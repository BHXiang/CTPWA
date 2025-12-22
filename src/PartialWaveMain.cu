// #include <pybind11/pybind11.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <omp.h>
#include <chrono>
#include <random>
#include <fstream>
#include <torch/extension.h>
#include <map>

#include <AmpGen.cuh>
#include <ComputeNLL.cuh>
#include <ComputeGrad.cuh>
#include <ComputeWeight.cuh>
#include <Config.cuh>

#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>

//////////////////////////////////////////////
struct ChainInfo
{
	std::string name;
	std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>> intermediate_resonance_map;
	std::vector<std::vector<Particle>> intermediate_combs;
};

class NLLFunction : public torch::autograd::Function<NLLFunction>
{
public:
	static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor &vector, int n_gls_, int n_polar_, const cuComplex *data_fix_, int data_length, const cuComplex *phsp_fix_, int phsp_length, const cuComplex *bkg_fix_, int bkg_length, std::vector<std::pair<int, int>> conjugate_pairs_)
	{
		TORCH_CHECK(vector.is_cuda(), "[NLLForward] vector must be on CUDA");
		TORCH_CHECK(vector.dtype() == c10::kComplexFloat, "[NLLForward] vector must be complex64");

		// 获取当前设备并设置
		const int target_dev = vector.get_device();
		torch::Device dev(torch::kCUDA, target_dev);

		// 延长vector以处理共轭对
		torch::Tensor extended_vector = extendVectorWithConjugates(vector, conjugate_pairs_, dev);
		const int extended_n_gls = extended_vector.numel();

		// 后续逻辑（MC因子计算等）
		cuComplex *d_B = nullptr;
		double *d_mc_amp = nullptr;
		cudaMalloc(&d_B, phsp_length * sizeof(cuComplex));
		cudaMalloc(&d_mc_amp, sizeof(double));

		// computeSingleResult(phsp_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_B, d_mc_amp, phsp_length, extended_n_gls);
		computePHSPfactor(phsp_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_B, d_mc_amp, phsp_length, extended_n_gls);

		double h_phsp_factor;
		cudaMemcpy(&h_phsp_factor, d_mc_amp, sizeof(double), cudaMemcpyDeviceToHost);
		h_phsp_factor = h_phsp_factor / static_cast<double>(phsp_length / n_polar_);

		// NLL计算
		cuComplex *d_S = nullptr;
		cuComplex *d_Q = nullptr;
		double *d_data_nll = nullptr;
		const int Q_numel = data_length / n_polar_;
		cudaMalloc(&d_S, data_length * sizeof(cuComplex));
		cudaMalloc(&d_Q, Q_numel * sizeof(cuComplex));
		cudaMalloc(&d_data_nll, sizeof(double));

		computeNll(data_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_S, d_Q, d_data_nll, data_length, extended_n_gls, n_polar_, h_phsp_factor);

		double h_data_nll;
		cudaMemcpy(&h_data_nll, d_data_nll, sizeof(double), cudaMemcpyDeviceToHost);

		// bkg部分
		cuComplex *d_bkg_S = nullptr;
		cuComplex *d_bkg_Q = nullptr;
		double *d_bkg_nll = nullptr;
		const int bkg_Q_numel = bkg_length / n_polar_;
		cudaMalloc(&d_bkg_S, bkg_length * sizeof(cuComplex));
		cudaMalloc(&d_bkg_Q, bkg_Q_numel * sizeof(cuComplex));
		cudaMalloc(&d_bkg_nll, sizeof(double));
		double h_bkg_nll = 0.0;
		if (bkg_fix_ != nullptr && bkg_length > 0)
		{
			computeNll(bkg_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_bkg_S, d_bkg_Q, d_bkg_nll, bkg_length, extended_n_gls, n_polar_, h_phsp_factor);

			cudaMemcpy(&h_bkg_nll, d_bkg_nll, sizeof(double), cudaMemcpyDeviceToHost);
		}

		// 保存反向传播变量
		ctx->saved_data["target_dev"] = target_dev;
		ctx->saved_data["n_polar"] = n_polar_;
		ctx->saved_data["h_phsp_factor"] = h_phsp_factor * static_cast<double>(phsp_length / n_polar_);
		ctx->saved_data["n_gls"] = n_gls_;
		ctx->saved_data["extended_n_gls"] = extended_n_gls;
		ctx->saved_data["data_length"] = data_length;
		ctx->saved_data["phsp_length"] = phsp_length;
		ctx->saved_data["bkg_length"] = bkg_length;

		// 保存共轭对信息
		torch::Tensor conjugate_pairs_tensor = torch::empty({static_cast<int64_t>(conjugate_pairs_.size()), 2}, torch::kInt32);
		auto conjugate_pairs_accessor = conjugate_pairs_tensor.accessor<int, 2>();
		for (size_t i = 0; i < conjugate_pairs_.size(); ++i)
		{
			conjugate_pairs_accessor[i][0] = conjugate_pairs_[i].first;
			conjugate_pairs_accessor[i][1] = conjugate_pairs_[i].second;
		}
		ctx->saved_data["conjugate_pairs"] = conjugate_pairs_tensor;

		// 保存显存指针
		ctx->saved_data["data_fix_ptr"] = reinterpret_cast<int64_t>(data_fix_);
		ctx->saved_data["phsp_fix_ptr"] = reinterpret_cast<int64_t>(phsp_fix_);
		ctx->saved_data["bkg_fix_ptr"] = reinterpret_cast<int64_t>(bkg_fix_);
		ctx->saved_data["d_B_ptr"] = reinterpret_cast<int64_t>(d_B);
		ctx->saved_data["d_S_ptr"] = reinterpret_cast<int64_t>(d_S);
		ctx->saved_data["d_Q_ptr"] = reinterpret_cast<int64_t>(d_Q);
		ctx->saved_data["d_bkg_S_ptr"] = reinterpret_cast<int64_t>(d_bkg_S);
		ctx->saved_data["d_bkg_Q_ptr"] = reinterpret_cast<int64_t>(d_bkg_Q);
		ctx->saved_data["d_bkg_nll_ptr"] = reinterpret_cast<int64_t>(d_bkg_nll);

		ctx->save_for_backward({vector, extended_vector});

		// 释放临时内存
		cudaFree(d_mc_amp);

		return torch::tensor({h_data_nll - h_bkg_nll}, torch::kDouble).to(dev);
	}

	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, const torch::autograd::tensor_list &grad_outputs)
	{
		const int target_dev = ctx->saved_data["target_dev"].toInt();

		// std::cout << "debug: " << __LINE__ << std::endl;

		// 从 saved_data 获取参数
		const int n_polar = ctx->saved_data["n_polar"].toInt();
		const double h_phsp_factor = ctx->saved_data["h_phsp_factor"].toDouble();
		const int n_gls = ctx->saved_data["n_gls"].toInt();
		const int extended_n_gls = ctx->saved_data["extended_n_gls"].toInt();
		const int data_length = ctx->saved_data["data_length"].toInt();
		const int phsp_length = ctx->saved_data["phsp_length"].toInt();
		const int bkg_length = ctx->saved_data["bkg_length"].toInt();

		// // 获取共轭对信息
		auto conjugate_pairs_tensor = ctx->saved_data["conjugate_pairs"].toTensor();
		std::vector<std::pair<int, int>> conjugate_pairs;
		auto conjugate_pairs_accessor = conjugate_pairs_tensor.accessor<int, 2>();
		for (int64_t i = 0; i < conjugate_pairs_tensor.size(0); ++i)
		{
			conjugate_pairs.push_back({conjugate_pairs_accessor[i][0], conjugate_pairs_accessor[i][1]});
		}

		// 从 saved_data 获取显存指针
		cuComplex *d_B = reinterpret_cast<cuComplex *>(ctx->saved_data["d_B_ptr"].toInt());
		cuComplex *data_fix = reinterpret_cast<cuComplex *>(ctx->saved_data["data_fix_ptr"].toInt());
		cuComplex *phsp_fix = reinterpret_cast<cuComplex *>(ctx->saved_data["phsp_fix_ptr"].toInt());
		cuComplex *d_S = reinterpret_cast<cuComplex *>(ctx->saved_data["d_S_ptr"].toInt());
		cuComplex *d_Q = reinterpret_cast<cuComplex *>(ctx->saved_data["d_Q_ptr"].toInt());

		// // 获取背景数据的指针（如果存在）
		// if (bkg_length > 0)
		// {
		cuComplex *bkg_fix = reinterpret_cast<cuComplex *>(ctx->saved_data["bkg_fix_ptr"].toInt());
		cuComplex *d_bkg_S = reinterpret_cast<cuComplex *>(ctx->saved_data["d_bkg_S_ptr"].toInt());
		cuComplex *d_bkg_Q = reinterpret_cast<cuComplex *>(ctx->saved_data["d_bkg_Q_ptr"].toInt());
		// }

		// 获取保存的变量
		const auto saved = ctx->get_saved_variables();
		const auto &original_vector = saved[0];
		const auto &extended_vector = saved[1];

		// 计算扩展向量的梯度
		cuComplex *d_extended_grad = nullptr;
		cudaMalloc(&d_extended_grad, extended_n_gls * sizeof(cuComplex));

		cublasHandle_t cublas_handle;
		cublasCreate(&cublas_handle);
		compute_gradient(data_fix, phsp_fix, d_S, d_Q, d_B, h_phsp_factor, extended_n_gls, data_length / n_polar, n_polar, phsp_length, d_extended_grad, cublas_handle);

		// 如果有背景数据，减去背景NLL的梯度
		if (bkg_fix != nullptr && bkg_length > 0)
		{
			cuComplex *d_bkg_extended_grad = nullptr;
			cudaMalloc(&d_bkg_extended_grad, extended_n_gls * sizeof(cuComplex));

			// 初始化背景梯度为0
			cudaMemset(d_bkg_extended_grad, 0, extended_n_gls * sizeof(cuComplex));

			// 计算背景NLL的梯度
			compute_gradient(bkg_fix, phsp_fix, d_bkg_S, d_bkg_Q, d_B, h_phsp_factor,
							 extended_n_gls, bkg_length / n_polar, n_polar,
							 phsp_length, d_bkg_extended_grad, cublas_handle);

			// 从数据梯度中减去背景梯度：∇L = ∇(data_nll) - ∇(bkg_nll)
			// 使用cublas的向量减法操作
			const cuComplex minus_one = make_cuComplex(-1.0f, 0.0f);
			cublasCaxpy(cublas_handle, extended_n_gls,
						&minus_one, d_bkg_extended_grad, 1,
						d_extended_grad, 1);

			cudaFree(d_bkg_extended_grad);
			cudaFree(d_bkg_S);
			cudaFree(d_bkg_Q);
		}

		// 		// 输出d_extended_grad以供调试
		// torch::Tensor debug_extended_grad = torch::empty({extended_n_gls}, torch::kComplexFloat).to(original_vector.device());
		// cudaMemcpy(debug_extended_grad.data_ptr(), d_extended_grad,
		// 		   extended_n_gls * sizeof(cuComplex),
		// 		   cudaMemcpyDeviceToDevice);
		// std::cout << "Debug extended_grad: " << debug_extended_grad << std::endl;

		// 将扩展梯度的共轭部分合并回原始梯度
		torch::Tensor extended_grad = torch::empty({extended_n_gls}, torch::kComplexFloat).to(original_vector.device());
		cudaMemcpy(extended_grad.data_ptr(), d_extended_grad,
				   extended_n_gls * sizeof(cuComplex),
				   cudaMemcpyDeviceToDevice);

		torch::Tensor grad_vector = mergeGradientsWithConjugates(extended_grad, conjugate_pairs, original_vector.numel());

		// 清理内存
		cudaFree(d_extended_grad);
		cudaFree(d_B);
		cudaFree(d_S);
		cudaFree(d_Q);
		cublasDestroy(cublas_handle);

		return {grad_vector, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
	}

private:
	static torch::Tensor extendVectorWithConjugates(const torch::Tensor &vector,
													const std::vector<std::pair<int, int>> &conjugate_pairs,
													const torch::Device &device)
	{
		const int original_size = vector.numel();
		int extended_size = original_size;

		// 计算需要扩展的大小
		for (const auto &pair : conjugate_pairs)
		{
			extended_size = std::max(extended_size, std::max(pair.first, pair.second) + 1);
		}

		if (extended_size == original_size)
		{
			return vector.clone();
		}

		// 创建扩展后的向量
		torch::Tensor extended_vector = torch::zeros({extended_size}, torch::kComplexFloat).to(device);

		// 复制原始向量到扩展向量
		extended_vector.slice(0, 0, original_size) = vector;

		// 设置共轭对
		for (const auto &pair : conjugate_pairs)
		{
			int source_idx = pair.first;
			int conjugate_idx = pair.second;

			if (source_idx < original_size && conjugate_idx < extended_size)
			{
				// 获取源元素的共轭
				torch::Tensor source_val = extended_vector[source_idx];
				// extended_vector[conjugate_idx] = torch::conj(source_val);
				extended_vector[conjugate_idx] = -1.0 * source_val;
			}
		}

		return extended_vector;
	}

	static torch::Tensor mergeGradientsWithConjugates(const torch::Tensor &extended_grad,
													  const std::vector<std::pair<int, int>> &conjugate_pairs,
													  int original_size)
	{
		torch::Tensor grad_vector = torch::zeros({original_size}, torch::kComplexFloat).to(extended_grad.device());

		// 复制直接梯度
		grad_vector = extended_grad.slice(0, 0, original_size);

		// 合并共轭对的梯度
		for (const auto &pair : conjugate_pairs)
		{
			int source_idx = pair.first;
			int conjugate_idx = pair.second;

			if (source_idx < original_size && conjugate_idx < extended_grad.numel())
			{
				// 对于共轭关系：y = conj(x)，梯度关系：dy/dx = conj(dL/dy)
				torch::Tensor conjugate_grad = extended_grad[conjugate_idx];
				// grad_vector[source_idx] = grad_vector[source_idx] + torch::conj(conjugate_grad);
				grad_vector[source_idx] = grad_vector[source_idx] - conjugate_grad;
			}
		}

		return grad_vector;
	}
};

////////////////////////////////////////
std::map<std::string, std::vector<LorentzVector>> readMomentaFromDat(
	const std::string &filename,
	const std::vector<std::string> &particleNames,
	int nEvents = -1)
{

	std::map<std::string, std::vector<LorentzVector>> finalMomenta;

	// 初始化粒子容器
	for (const auto &name : particleNames)
	{
		finalMomenta[name] = std::vector<LorentzVector>();
	}

	std::ifstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error: Cannot open file " << filename << std::endl;
		return finalMomenta;
	}

	std::string line;
	int eventCount = 0;
	int lineCount = 0;
	int particlesPerEvent = particleNames.size();

	while (std::getline(file, line))
	{
		if (line.empty())
			continue;

		std::istringstream iss(line);
		double E, px, py, pz;

		if (iss >> E >> px >> py >> pz)
		{
			// 根据行号确定粒子类型
			int particleIndex = lineCount % particlesPerEvent;
			const std::string &particleName = particleNames[particleIndex];

			finalMomenta[particleName].emplace_back(E, px, py, pz);
			lineCount++;

			// 每读完一组粒子表示完成一个事件
			if (particleIndex == particlesPerEvent - 1)
			{
				eventCount++;

				// 如果指定了事件数并且已达到，则停止读取
				if (nEvents > 0 && eventCount >= nEvents)
				{
					break;
				}
			}
		}
		else
		{
			std::cerr << "Warning: Invalid line format: " << line << std::endl;
		}
	}

	file.close();

	// std::cout << "Successfully read " << eventCount << " events from " << filename << std::endl;
	return finalMomenta;
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
class analysis
{
public:
	analysis(const std::string &config_file = "config.yml") : config_parser_(config_file), n_gls_(0), n_polar_(0), data_fix_(nullptr), data_length(0), phsp_fix_(nullptr), phsp_length(0), bkg_fix_(nullptr), bkg_length(0)
	{
		initialize();
	}

	// 析构函数，用于释放 CUDA 内存
	~analysis()
	{
		if (data_fix_ != nullptr)
		{
			cudaFree(data_fix_);
			data_fix_ = nullptr;
		}
		if (phsp_fix_ != nullptr)
		{
			cudaFree(phsp_fix_);
			phsp_fix_ = nullptr;
		}
		if (bkg_fix_ != nullptr)
		{
			cudaFree(bkg_fix_);
			bkg_fix_ = nullptr;
		}
	}

	torch::Tensor getNLL(torch::Tensor &vector)
	{
		return NLLFunction::apply(vector, n_gls_, n_polar_, data_fix_, data_length, phsp_fix_, phsp_length, bkg_fix_, bkg_length, conjugate_pairs_);
		// return NLLFunction::apply(vector, n_gls_, n_polar_, data_fix_, data_length, phsp_fix_, phsp_length, bkg_fix_, bkg_length);
	}

	int getNVector() const
	{
		return n_gls_ - conjugate_pairs_.size();
	}

	void writeWeightFile(torch::Tensor &vector, const std::string &filename)
	{
		TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
		TORCH_CHECK(vector.dtype() == torch::kComplexFloat, "vector must be complex128");

		const int original_size = vector.numel();
		int extended_size = original_size;

		// const int target_dev = vector.get_device();
		torch::Device dev(torch::kCUDA, vector.get_device());

		// 计算需要扩展的大小
		for (const auto &pair : conjugate_pairs_)
		{
			extended_size = std::max(extended_size, std::max(pair.first, pair.second) + 1);
		}

		// if (extended_size == original_size)
		// {
		// 	return vector.clone();
		// }

		// 创建扩展后的向量
		torch::Tensor extended_vector = torch::zeros({extended_size}, torch::kComplexFloat).to(dev);

		// 复制原始向量到扩展向量
		extended_vector.slice(0, 0, original_size) = vector;

		// 设置共轭对
		for (const auto &pair : conjugate_pairs_)
		{
			int source_idx = pair.first;
			int conjugate_idx = pair.second;

			// std::cout << "Conjugate pair: " << source_idx << " <-> " << conjugate_idx << std::endl;

			if (source_idx < original_size && conjugate_idx < extended_size)
			{
				// 获取源元素的共轭
				torch::Tensor source_val = extended_vector[source_idx];
				// extended_vector[conjugate_idx] = torch::conj(source_val);
				extended_vector[conjugate_idx] = -1.0 * source_val;
			}
		}

		////////////////////////////////////////////////////////////////////////////
		// std::cout << "Extended vector: " << extended_vector.cpu() << std::endl;
		////////////////////////////////////////////////////////////////////////////

		const int target_dev = vector.get_device();
		cudaSetDevice(target_dev);

		const int n_events = phsp_length / n_polar_; // 事件数量
		double *d_final_result;
		double *d_partial_result;
		double *d_partial_sum;

		// 分配设备内存
		cudaMalloc(&d_final_result, n_events * sizeof(double));
		int npartials = nSLvectors_.size();
		cudaMalloc(&d_partial_result, n_events * npartials * sizeof(double));
		cudaMalloc(&d_partial_sum, npartials * sizeof(double));

		// 分配nSLvectors_的设备内存
		int *d_nSLvectors;
		cudaMalloc(&d_nSLvectors, nSLvectors_.size() * sizeof(int));
		cudaMemcpy(d_nSLvectors, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);
		double *d_total_integral;
		cudaMalloc(&d_total_integral, sizeof(double));
		cudaMemset(d_total_integral, 0, sizeof(double));

		// 计算权重
		computeWeightResult(phsp_fix_, reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()), d_final_result, d_total_integral, d_partial_result, d_partial_sum, d_nSLvectors, npartials, n_events, n_gls_, n_polar_);

		double *h_total_results = new double[n_events];
		cudaMemcpy(h_total_results, d_final_result, n_events * sizeof(double), cudaMemcpyDeviceToHost);
		double *h_partial_results = new double[n_events * npartials];
		cudaMemcpy(h_partial_results, d_partial_result, n_events * npartials * sizeof(double), cudaMemcpyDeviceToHost);
		double *h_partial_sums = new double[npartials];
		cudaMemcpy(h_partial_sums, d_partial_sum, npartials * sizeof(double), cudaMemcpyDeviceToHost);
		double h_phsp_integral;
		cudaMemcpy(&h_phsp_integral, d_total_integral, sizeof(double), cudaMemcpyDeviceToHost);

		// 写入文件
		// std::ofstream outfile(filename);
		// if (outfile.is_open())
		// {
		// 	// for (const auto &weight : h_row_results)
		// 	for (int i = 0; i < n_events; ++i)
		// 	{
		// 		outfile << h_total_results[i] << std::endl;
		// 	}
		// 	outfile.close();
		// 	std::cout << "Weights written to " << filename << std::endl;
		// }
		// else
		// {
		// 	std::cerr << "Unable to open file: " << filename << std::endl;
		// }

		int dataIntegral = data_length / n_polar_;

		// 创建 ROOT 文件
		TFile *rootFile = new TFile(filename.c_str(), "RECREATE");

		// 创建 data 的 TTree
		if (!Vp4_data_.empty())
		{
			TTree *dataTree = new TTree("t_data", "Data events with four-momenta");

			// 为每个粒子创建 TLorentzVector 分支
			std::map<std::string, TLorentzVector> data_vectors;
			for (const auto &particle : Vp4_data_)
			{
				data_vectors[particle.first] = TLorentzVector();
				dataTree->Branch(particle.first.c_str(), &data_vectors[particle.first]);
			}

			// 填充 data tree
			int n_data_events = Vp4_data_.begin()->second.size();
			for (int i = 0; i < n_data_events; ++i)
			{
				for (const auto &particle : Vp4_data_)
				{
					const auto &lv = particle.second[i];
					data_vectors[particle.first].SetPxPyPzE(lv.Px, lv.Py, lv.Pz, lv.E);
				}
				dataTree->Fill();
			}
			dataTree->Write();
			delete dataTree;
		}

		// 创建 phsp 的 TTree
		if (!Vp4_phsp_.empty())
		{
			TTree *phspTree = new TTree("t_phsp", "Phase space events with weights");

			// 为每个粒子创建 TLorentzVector 分支
			std::map<std::string, TLorentzVector> phsp_vectors;
			for (const auto &particle : Vp4_phsp_)
			{
				phsp_vectors[particle.first] = TLorentzVector();
				phspTree->Branch(particle.first.c_str(), &phsp_vectors[particle.first]);
			}

			// 添加权重分支
			double total_weight;
			std::vector<double> partial_weights(npartials);
			phspTree->Branch("totalweight", &total_weight);

			// 为每个部分波创建分支
			for (int i = 0; i < npartials; ++i)
			{
				std::string branch_name = "partialweight_" + std::to_string(i);
				phspTree->Branch(branch_name.c_str(), &partial_weights[i]);
			}

			// 填充 phsp tree
			for (int i = 0; i < n_events; ++i)
			{
				// 设置四动量
				for (const auto &particle : Vp4_phsp_)
				{
					const auto &lv = particle.second[i];
					phsp_vectors[particle.first].SetPxPyPzE(lv.Px, lv.Py, lv.Pz, lv.E);
				}

				// 设置权重
				total_weight = h_total_results[i] / h_phsp_integral * static_cast<double>(dataIntegral);
				// std::cout << "Event " << i << ": Total Weight = " << total_weight << std::endl;
				for (int j = 0; j < npartials; ++j)
				{
					partial_weights[j] = h_partial_results[i * npartials + j] * static_cast<double>(dataIntegral) / h_phsp_integral;
					// std::cout << "  Partial Weight " << j << " = " << partial_weights[j] << std::endl;
				}

				phspTree->Fill();
			}

			// write legend branches
			phspTree->Branch("legends", &legends_);
			phspTree->Fill();

			phspTree->Write();
			delete phspTree;
		}

		// 创建 bkg 的 TTree
		if (!Vp4_bkg_.empty())
		{
			TTree *bkgTree = new TTree("t_bkg", "Background events with four-momenta");

			// 为每个粒子创建 TLorentzVector 分支
			std::map<std::string, TLorentzVector> bkg_vectors;
			for (const auto &particle : Vp4_bkg_)
			{
				bkg_vectors[particle.first] = TLorentzVector();
				bkgTree->Branch(particle.first.c_str(), &bkg_vectors[particle.first]);
			}

			// 填充 bkg tree
			int n_bkg_events = Vp4_bkg_.begin()->second.size();
			for (int i = 0; i < n_bkg_events; ++i)
			{
				for (const auto &particle : Vp4_bkg_)
				{
					const auto &lv = particle.second[i];
					bkg_vectors[particle.first].SetPxPyPzE(lv.Px, lv.Py, lv.Pz, lv.E);
				}
				bkgTree->Fill();
			}
			bkgTree->Write();
			delete bkgTree;
		}

		// TTree *legend = new TTree("legends", "Amplitude Legends");
		// legend->Branch("legend", &legends_);
		// legend->Fill();
		// legend->Write();
		// delete legend;

		// 关闭 ROOT 文件
		rootFile->Close();
		delete rootFile;

		// std::cout << "Data written to ROOT file: " << filename << std::endl;

		// 释放设备内存
		// cudaFree(d_row_results);
		cudaFree(d_final_result);
		cudaFree(d_partial_result);
		cudaFree(d_nSLvectors);
		delete[] h_total_results;
		delete[] h_partial_results;
	}

	torch::Tensor getDataTensor() const
	{
		std::vector<std::complex<double>> host_array(data_length * n_gls_);
		cudaMemcpy(host_array.data(), data_fix_, data_length * n_gls_ * sizeof(cuComplex), cudaMemcpyDeviceToHost);

		// std::cout << "debug " << host_array[0] << std::endl;

		torch::Tensor output = torch::empty({data_length * n_gls_}, dtype(torch::kComplexFloat));
		output.copy_(torch::from_blob(host_array.data(), {data_length * n_gls_}, torch::kComplexFloat));

		// cudaFree(host_array);

		return output;
	}

	torch::Tensor getPhspTensor() const
	{
		std::vector<std::complex<double>> host_array(phsp_length * n_gls_);
		cudaMemcpy(host_array.data(), phsp_fix_, phsp_length * n_gls_ * sizeof(cuComplex), cudaMemcpyDeviceToHost);

		// std::cout << "debug " << host_array[0] << std::endl;

		torch::Tensor output = torch::empty({phsp_length * n_gls_}, dtype(torch::kComplexFloat));
		output.copy_(torch::from_blob(host_array.data(), {phsp_length * n_gls_}, torch::kComplexFloat));

		// cudaFree(host_array);

		return output;
	}

	torch::Tensor getBkgTensor() const
	{
		std::vector<std::complex<double>> host_array(bkg_length);
		cudaMemcpy(host_array.data(), bkg_fix_, bkg_length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

		// std::cout << "debug " << host_array[0] << std::endl;

		torch::Tensor output = torch::empty({bkg_length}, dtype(torch::kComplexFloat));
		output.copy_(torch::from_blob(host_array.data(), {bkg_length}, torch::kComplexFloat));

		// cudaFree(host_array);

		return output;
	}

	std::vector<std::pair<int, int>> getConjugatePairs() const
	{
		return conjugate_pairs_;
	}

	std::vector<std::string> getAmplitudeNames() const
	{
		return amplitude_names_;
	}

private:
	int n_gls_;
	int n_polar_ = 1;
	std::vector<int> nSLvectors_;
	cuComplex *data_fix_;
	int data_length;
	cuComplex *phsp_fix_;
	int phsp_length;
	cuComplex *bkg_fix_;
	int bkg_length;

	// 四动量数据
	std::map<std::string, std::vector<LorentzVector>> Vp4_data_;
	std::map<std::string, std::vector<LorentzVector>> Vp4_phsp_;
	std::map<std::string, std::vector<LorentzVector>> Vp4_bkg_;

	// amplitude 信息
	std::vector<std::pair<int, int>> conjugate_pairs_;
	std::vector<std::string> amplitude_names_;
	std::vector<std::string> legends_;

	// config 初始化
	ConfigParser config_parser_;
	std::vector<Particle> particles_;
	std::unordered_map<std::string, Resonance> resonances_;
	int n_amplitudes_ = 0;
	std::vector<ChainInfo> chains_info_;

	void initialize(std::string config_file = "config.yml")
	{
		// 读取配置文件
		// std::string config_file = "config.yml";
		// config_parser_(config_file);
		std::cout << "Reading config file: " << config_file << std::endl;
		std::cout << "  Particles: " << config_parser_.getParticles().size() << std::endl;
		std::cout << "  Decay chains: " << config_parser_.getDecayChains().size() << std::endl;
		std::cout << "  Resonances: " << config_parser_.getResonances().size() << std::endl;
		std::cout << "  Conjugate pairs: " << config_parser_.getConjugatePairs().size() << std::endl;

		// 初始化粒子信息
		initializeParticles();
		// 初始化极化状态
		initializePolarization();
		// 初始化衰变链
		initializeDecayChains();

		// // 创建振幅计算器，自动读取配置
		// AmplitudeCalculator calculator(config_file);

		// // 获取配置信息
		// const auto &config_parser = calculator.getConfigParser();
		const auto &data_files = config_parser_.getDataFiles();
		const auto &dat_order = config_parser_.getDatOrder();

		legends_ = config_parser_.getCustomLegends();
		n_gls_ = n_amplitudes_;

		// 计算相空间振幅
		std::cout << "Calculating phase space amplitudes..." << std::endl;
		std::cout << "Reading phase space data..." << std::endl;
		Vp4_phsp_ = readMomentaFromDat(data_files.at("phsp")[0], dat_order);
		std::cout << "Phase space events: " << Vp4_phsp_.begin()->second.size() << std::endl;
		phsp_fix_ = calculateAmplitudes(Vp4_phsp_);
		phsp_length = Vp4_phsp_.begin()->second.size() * n_polar_;

		// 计算数据振幅
		std::cout << "Calculating data amplitudes..." << std::endl;
		std::cout << "Reading data events..." << std::endl;
		Vp4_data_ = readMomentaFromDat(data_files.at("data")[0], dat_order);
		std::cout << "Data events: " << Vp4_data_.begin()->second.size() << std::endl;
		data_fix_ = calculateAmplitudes(Vp4_data_);
		data_length = Vp4_data_.begin()->second.size() * n_polar_;

		// 计算本底振幅
		if (data_files.count("bkg") > 0)
		{
			std::cout << "Calculating background amplitudes..." << std::endl;
			std::cout << "Reading background data..." << std::endl;
			Vp4_bkg_ = readMomentaFromDat(data_files.at("bkg")[0], dat_order);
			std::cout << "Background events: " << Vp4_bkg_.begin()->second.size() << std::endl;
			bkg_fix_ = calculateAmplitudes(Vp4_bkg_);
			bkg_length = Vp4_bkg_.begin()->second.size() * n_polar_;
		}

		// // 计算相空间振幅
		// // std::cout << "Calculating phase space amplitudes..." << std::endl;
		// phsp_fix_ = calculator.calculateAmplitudes(Vp4_phsp_, 1);
		// std::cout << "Reading phase space data..." << std::endl;
		// std::cout << "Phase space events: " << Vp4_phsp_.begin()->second.size() << std::endl;
		// n_gls_ = calculator.getNAmplitudes();
		// n_polar_ = calculator.getNPolarization(); // 假设每个事件有3个极化状态
		// phsp_length = Vp4_phsp_.begin()->second.size() * n_polar_;

		// // std::cout << "Number of amplitudes (n_gls_): " << n_gls_ << std::endl;

		// // 读取数据
		// std::cout << "Reading data..." << std::endl;
		// Vp4_data_ = readMomentaFromDat(data_files.at("data")[0], dat_order);
		// std::cout << "Data events: " << Vp4_data_.begin()->second.size() << std::endl;

		// // 计算数据振幅
		// // std::cout << "Calculating data amplitudes..." << std::endl;
		// data_fix_ = calculator.calculateAmplitudes(Vp4_data_, 0);
		// data_length = Vp4_data_.begin()->second.size() * n_polar_;

		// // 读取背景数据
		// if (data_files.count("bkg") > 0)
		// {
		// 	std::cout << "Reading background data..." << std::endl;
		// 	Vp4_bkg_ = readMomentaFromDat(data_files.at("bkg")[0], dat_order);
		// 	std::cout << "Background events: " << Vp4_bkg_.begin()->second.size() << std::endl;
		// 	// 计算背景振幅
		// 	// std::cout << "Calculating background amplitudes..." << std::endl;
		// 	bkg_fix_ = calculator.calculateAmplitudes(Vp4_bkg_, 0);
		// 	bkg_length = Vp4_bkg_.begin()->second.size() * n_polar_;
		// }

		// // 获取振幅名称和共轭对信息
		// amplitude_names_ = calculator.getAmplitudeNames();
		// conjugate_pairs_ = calculator.getConjugatePairIndices();
		// nSLvectors_ = calculator.getNSLVectors();

		std::cout << "Number of partial waves (n_gls_): " << n_gls_ << std::endl;
		std::cout << "Number of amplitude names: " << amplitude_names_.size() << std::endl;
		std::cout << "Number of constraints: " << conjugate_pairs_.size() << std::endl;
		std::cout << "Initialization complete." << std::endl;
	}

	void initializeParticles()
	{
		const auto &config_particles = config_parser_.getParticles();
		for (const auto &particle_config : config_particles)
		{
			particles_.push_back(particle_config);
		}
	}

	void initializePolarization()
	{
		n_polar_ = 1;

		for (const auto &particle : particles_)
		{
			if (particle.spin > 0)
			{
				n_polar_ *= (2 * particle.spin + 1);
			}
		}

		std::cout << "Total polarization states (n_polar_): " << n_polar_ << std::endl;
	}

	void initializeDecayChains()
	{
		auto chains = config_parser_.getDecayChains();

		const auto &config_resonances = config_parser_.getResonances();

		// 获取总振幅长度
		for (const auto &chain : chains)
		{
			// std::cout << "Processing decay chain: " << chain.name << std::endl;

			ChainInfo chain_info;
			chain_info.name = chain.name;

			std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>> intermediate_resonance_map;
			std::vector<std::vector<Particle>> intermediate_particles;
			for (const auto &res_chain : chain.resonance_chains)
			{
				// std::cout << "  Intermediate: " << res_chain.intermediate << std::endl;
				std::vector<Particle> particles;
				for (const auto &spin_chain : res_chain.spin_chains)
				{
					// std::cout << "    Spin-Parity options: ";

					Particle intermediate_particle = {res_chain.intermediate, static_cast<int>(spin_chain.spin_parity[0]), static_cast<int>(spin_chain.spin_parity[1]), -1};

					std::vector<Resonance> resonance_list;
					// std::cout << " Resonance: " << std::endl;
					for (const auto &resonance_name : spin_chain.resonances)
					{
						// std::cout << resonance_name << std::endl;

						for (const auto &[name, res_config] : config_resonances)
						{
							if (name == resonance_name)
							{
								if (res_config.J == spin_chain.spin_parity[0] && spin_chain.spin_parity[1])
								{
									resonance_list.emplace_back(name, res_chain.intermediate, intermediate_particle.spin, intermediate_particle.parity, res_config.type, res_config.parameters);
									// std::cout << "      Added resonance: " << name << " J: " << res_config.J << " P: " << res_config.P << std::endl;
								}
								else
								{
									std::cout << "      Skipped resonance (J,P mismatch): " << name << " J: " << res_config.J << " P: " << res_config.P << std::endl;
								}
							}
						}
					}
					// std::cout << std::endl;

					std::pair<std::string, std::vector<int>> key = {res_chain.intermediate, {spin_chain.spin_parity[0], spin_chain.spin_parity[1]}};
					intermediate_resonance_map[key] = resonance_list;
					particles.push_back(intermediate_particle);
				}
				intermediate_particles.push_back(particles);
			}

			chain_info.intermediate_resonance_map = intermediate_resonance_map;

			std::vector<std::vector<Particle>> intermediate_combs = {{}};
			for (const auto &particleList : intermediate_particles)
			{
				std::vector<std::vector<Particle>> temp;
				for (const auto &comb : intermediate_combs)
				{
					for (const auto &particle : particleList)
					{
						std::vector<Particle> new_res = comb;
						// std::cout << " Adding particle to combination: " << particle.name << " J: " << particle.spin << " P: " << particle.parity << std::endl;
						new_res.push_back(particle);
						temp.push_back(new_res);
					}
				}
				intermediate_combs = std::move(temp);
			}

			chain_info.intermediate_combs = intermediate_combs;

			for (auto comb : intermediate_combs)
			{
				AmpCasDecay cas(particles_);
				for (const auto &step : chain.decay_steps)
				{
					std::array<int, 3> spins = {0};
					std::array<int, 3> parities = {0};
					for (auto particle : particles_)
					{
						if (particle.name == step.mother)
						{
							// std::cout << "mother: " << particle.name << " " << particle.spin << " " << particle.parity << std::endl;
							spins[0] = particle.spin;
							parities[0] = particle.parity;
						}

						for (int i = 0; i < step.daughters.size(); i++)
						{
							if (particle.name == step.daughters[i])
							{
								// std::cout << "daugters: " << particle.name << " " << particle.spin << " " << particle.parity << std::endl;
								spins[i + 1] = particle.spin;
								parities[i + 1] = particle.parity;
							}
						}
					}
					for (auto res_jp : comb)
					{
						if (res_jp.name == step.mother)
						{
							// std::cout << "mother: " << res_jp.name << " " << res_jp.spin << std::endl;
							spins[0] = res_jp.spin;
							parities[0] = res_jp.parity;
						}

						for (int i = 0; i < step.daughters.size(); i++)
						{
							if (res_jp.name == step.daughters[i])
							{
								// std::cout << "daugters: " << res_jp.name << " " << res_jp.spin << " " << res_jp.parity << std::endl;
								spins[i + 1] = res_jp.spin;
								parities[i + 1] = res_jp.parity;
							}
						}
					}
					cas.addDecay(Amp2BD(spins, parities), step.mother, step.daughters[0], step.daughters[1]);

					// 输出decay chain结构
					std::cout << step.mother << "(" << spins[0];
					if (parities[0] == 1)
						std::cout << "+)";
					else if (parities[0] == -1)
						std::cout << "-)";
					std::cout << "->";
					for (int i = 0; i < step.daughters.size(); i++)
					{
						std::cout << step.daughters[i] << "(" << spins[i + 1];
						if (parities[i + 1] == 1)
							std::cout << "+)";
						else if (parities[i + 1] == -1)
							std::cout
								<< "-)";
					}
					std::cout << ", ";
				}

				auto slcombs = cas.getSLCombinations();
				std::cout << "SL:";
				for (auto slcomb : slcombs)
				{
					std::cout << "{";
					for (auto sl : slcomb)
					{
						std::cout << "(" << sl.S << ", " << sl.L << ")";
					}
					std::cout << "}";
				}
				std::cout << std::endl;

				std::vector<std::vector<std::pair<std::string, std::string>>> resonance_combinations = {{}};
				for (const auto &particle : comb)
				{
					std::pair<std::string, std::vector<int>> key = {particle.name, {particle.spin, particle.parity}};
					const auto &resonance_list = intermediate_resonance_map[key];

					std::vector<std::vector<std::pair<std::string, std::string>>> temp;
					for (const auto &res_comb : resonance_combinations)
					{
						for (const auto &resonance : resonance_list)
						{
							std::vector<std::pair<std::string, std::string>> new_res_comb = res_comb;
							new_res_comb.push_back({resonance.getTag(), resonance.getName()});
							temp.push_back(new_res_comb);
						}
					}
					resonance_combinations = std::move(temp);
				}

				std::cout << "Resonance: ";
				for (size_t k = 0; k < resonance_combinations.size(); ++k)
				{
					n_amplitudes_ += slcombs.size();
					nSLvectors_.push_back(slcombs.size());

					std::string res_name = chain.name;
					std::cout << "{ ";
					for (size_t j = 0; j < resonance_combinations[k].size(); ++j)
					{
						const auto &res_pair = resonance_combinations[k][j];
						res_name += "-" + res_pair.first + "-" + res_pair.second;

						std::cout << res_pair.second; // 共振态名称
						if (j < resonance_combinations[k].size() - 1)
							std::cout << ", ";
					}
					if (k < resonance_combinations.size() - 1)
						std::cout << " }, ";
					else
						std::cout << "}";

					for (const auto &slcomb : slcombs)
					{
						std::string full_name = res_name + "-" + "SL";
						for (const auto &sl : slcomb)
						{
							full_name += "_" + std::to_string(sl.S) + std::to_string(sl.L);
						}
						amplitude_names_.push_back(full_name);
					}
				}
				std::cout << std::endl;
			}

			chains_info_.push_back(chain_info);
		}

		// 设置约束条件
		auto conjugate_name = config_parser_.getConjugatePairs();
		std::vector<int> idx1;
		std::vector<int> idx2;
		for (const auto &pair : conjugate_name)
		{
			for (int i = 0; i < amplitude_names_.size(); ++i)
			{
				if (amplitude_names_[i].find(pair.first) != std::string::npos)
				{
					idx1.push_back(i);
				}
				if (amplitude_names_[i].find(pair.second) != std::string::npos)
				{
					idx2.push_back(i);
				}
			}
		}
		if (idx1.size() != idx2.size())
		{
			std::cerr << "Error: Mismatched conjugate pair sizes!" << std::endl;
		}
		else
		{
			for (int i = 0; i < idx1.size(); ++i)
			{
				// std::cout << "Conjugate pair: " << amplitude_names_[idx1[i]] << " <-> " << amplitude_names_[idx2[i]] << std::endl;
				if (idx1[i] < idx2[i])
				{
					conjugate_pairs_.emplace_back(idx1[i], idx2[i]);
				}
				else if (idx2[i] < idx1[i])
				{
					conjugate_pairs_.emplace_back(idx2[i], idx1[i]);
				}
			}
		}
	}

	cuComplex *calculateAmplitudes(const std::map<std::string, std::vector<LorentzVector>> &Vp4)
	{
		cuComplex *d_all_amplitudes = nullptr;
		const size_t n_events = Vp4.begin()->second.size();
		cudaMalloc(&d_all_amplitudes, n_events * n_amplitudes_ * n_polar_ * sizeof(cuComplex));

		// std::cout << "Calculating amplitudes for " << n_events << " events..." << std::endl;
		// std::cout << "polarization states: " << n_polar_ << std::endl;
		// std::cout << "total amplitudes: " << n_amplitudes_ << std::endl;

		auto chains = config_parser_.getDecayChains();
		int gls_index = 0;
		for (auto chain : chains)
		{
			ChainInfo chain_info;
			for (auto chaininfo : chains_info_)
			{
				if (chaininfo.name == chain.name)
				{
					chain_info = chaininfo;
					break;
				}
			}
			auto intermediate_resonance_map = chain_info.intermediate_resonance_map;
			auto intermediate_combs = chain_info.intermediate_combs;

			for (auto comb : intermediate_combs)
			{
				AmpCasDecay cas(particles_);
				for (const auto &step : chain.decay_steps)
				{
					std::array<int, 3> spins = {0};
					std::array<int, 3> parities = {0};
					for (auto particle : particles_)
					{
						if (particle.name == step.mother)
						{
							spins[0] = particle.spin;
							parities[0] = particle.parity;
						}

						for (int i = 0; i < step.daughters.size(); i++)
						{
							if (particle.name == step.daughters[i])
							{
								spins[i + 1] = particle.spin;
								parities[i + 1] = particle.parity;
							}
						}
					}
					for (auto res_jp : comb)
					{
						if (res_jp.name == step.mother)
						{
							spins[0] = res_jp.spin;
							parities[0] = res_jp.parity;
						}

						for (int i = 0; i < step.daughters.size(); i++)
						{
							if (res_jp.name == step.daughters[i])
							{
								spins[i + 1] = res_jp.spin;
								parities[i + 1] = res_jp.parity;
							}
						}
					}
					cas.addDecay(Amp2BD(spins, parities), step.mother, step.daughters[0], step.daughters[1]);
				}

				auto slcombs = cas.getSLCombinations();

				std::vector<std::vector<Resonance>> resonance_combinations = {{}};
				for (const auto &particle : comb)
				{
					std::pair<std::string, std::vector<int>> key = {particle.name, {particle.spin, particle.parity}};
					std::vector<Resonance> &resonance_list = intermediate_resonance_map[key];

					std::vector<std::vector<Resonance>> temp;
					for (const auto &res_comb : resonance_combinations)
					{
						for (const auto &resonance : resonance_list)
						{
							std::vector<Resonance> new_res_comb = res_comb;
							new_res_comb.push_back(resonance); // 直接添加 Resonance 对象
							temp.push_back(std::move(new_res_comb));
						}
					}
					resonance_combinations = std::move(temp);
				}

				cas.computeSLAmps(Vp4);
				int nSLcombs = cas.getNSLCombs();
				int nEvents = cas.getNEvents();

				for (const auto resonance : resonance_combinations)
				{
					cas.getAmps(d_all_amplitudes, resonance, gls_index);
					gls_index += 1;
				}
			}
		}

		return d_all_amplitudes;
	}
};

// 定义Python模块
PYBIND11_MODULE(ctpwa, m)
{
	m.doc() = "ctpwa";
	pybind11::class_<std::pair<int, int>>(m, "ConjugatePair")
		.def_readonly("first", &std::pair<int, int>::first)
		.def_readonly("second", &std::pair<int, int>::second);

	pybind11::class_<analysis>(m, "analysis")
		.def(pybind11::init<>())
		.def("getNLL", &analysis::getNLL)
		.def("getNVector", &analysis::getNVector)
		.def("writeWeightFile", &analysis::writeWeightFile)
		.def("getDataTensor", &analysis::getDataTensor)
		.def("getPhspTensor", &analysis::getPhspTensor)
		.def("getBkgTensor", &analysis::getBkgTensor)
		.def("getConjugatePairs", &analysis::getConjugatePairs)
		.def("getAmplitudeNames", &analysis::getAmplitudeNames);
}
