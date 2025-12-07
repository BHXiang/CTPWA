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
#include <yaml-cpp/yaml.h>

#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>

// namespace py = pybind11;

struct ResonanceConfig
{
	std::string name;
	int J;
	int P;
	std::string type;
	std::vector<double> parameters;
	std::string tex;
};

struct SpinChainConfig
{
	std::vector<int> spin_parity;
	std::vector<std::string> resonances;
};

struct ResonanceChainConfig
{
	std::string intermediate;
	std::vector<SpinChainConfig> spin_chains;
};

struct DecayStep
{
	std::string mother;
	std::vector<std::string> daughters;
};

struct DecayChainConfig
{
	std::string name;
	std::vector<DecayStep> decay_steps;
	std::vector<ResonanceChainConfig> resonance_chains;
	std::vector<std::string> legend_template;
};

class ConfigParser
{
public:
	ConfigParser(const std::string &config_file)
	{
		try
		{
			YAML::Node config = YAML::LoadFile(config_file);

			if (config["particles"])
			{
				parseParticles(config["particles"]);
			}

			if (config["data"])
			{
				parseData(config["data"]);
			}

			if (config["decay_chains"])
			{
				parseDecayChains(config["decay_chains"]);
			}

			if (config["resonances"])
			{
				parseResonances(config["resonances"]);
			}

			if (config["conjugate_pairs"])
			{
				parseConjugatePairs(config["conjugate_pairs"]);
			}
		}
		catch (const YAML::Exception &e)
		{
			std::cerr << "Error parsing config file: " << e.what() << std::endl;
			throw;
		}
	};

	const std::vector<Particle> &getParticles() const { return particles_; }
	const std::vector<DecayChainConfig> &getDecayChains() const { return decay_chains_; }
	const std::map<std::string, ResonanceConfig> &getResonances() const { return resonances_; }
	const std::vector<std::pair<std::string, std::string>> &getConjugatePairs() const { return conjugate_pairs_; }
	const std::map<std::string, std::vector<std::string>> &getDataFiles() const { return data_files_; }
	const std::vector<std::string> &getDatOrder() const { return dat_order_; }

	// 定制legend功能
	std::vector<std::string> getCustomLegends() const { return generateCustomLegends(); }
	std::map<std::string, std::vector<std::string>> getChainCustomLegends() const { return generateChainCustomLegends(); }

	// 生成Legend的函数
	std::string generateLegend(const std::vector<std::string> &particles) const
	{
		std::string legend;
		for (size_t i = 0; i < particles.size(); ++i)
		{
			const auto &particle_name = particles[i];

			// 首先检查是否是粒子
			auto particle_it = std::find_if(particles_.begin(), particles_.end(),
											[&](const Particle &p)
											{ return p.name == particle_name; });
			if (particle_it != particles_.end())
			{
				legend += particle_it->tex;
			}
			else
			{
				// 如果不是粒子，检查是否是共振态
				auto res_it = resonances_.find(particle_name);
				if (res_it != resonances_.end())
				{
					legend += res_it->second.tex;
				}
				else
				{
					// 如果都不是，使用原始名称作为字符输出
					legend += particle_name;
				}
			}

			if (i < particles.size() - 1)
			{
				legend += "";
			}
		}
		return legend;
	}

	// 获取单个粒子的Tex
	std::string getParticleTex(const std::string &name) const
	{
		auto particle_it = std::find_if(particles_.begin(), particles_.end(),
										[&](const Particle &p)
										{ return p.name == name; });
		if (particle_it != particles_.end())
		{
			return particle_it->tex;
		}
		return name; // 如果找不到，返回原始名称
	}

	// 获取单个共振态的Tex
	std::string getResonanceTex(const std::string &name) const
	{
		auto res_it = resonances_.find(name);
		if (res_it != resonances_.end())
		{
			return res_it->second.tex;
		}
		return name; // 如果找不到，返回原始名称
	}

	// 按decay chain顺序获取所有legend
	std::vector<std::string> getAllLegends() const
	{
		std::vector<std::string> legends;

		for (const auto &chain : decay_chains_)
		{
			// 为每个decay chain生成legend
			for (const auto &step : chain.decay_steps)
			{
				// 生成母粒子和子粒子的legend
				std::vector<std::string> particles_for_legend;
				particles_for_legend.push_back(step.mother);
				particles_for_legend.insert(particles_for_legend.end(),
											step.daughters.begin(), step.daughters.end());

				std::string legend = generateLegend(particles_for_legend);
				legends.push_back(legend);
			}

			// 为每个共振态链生成legend
			for (const auto &res_chain : chain.resonance_chains)
			{
				for (const auto &spin_chain : res_chain.spin_chains)
				{
					for (const auto &resonance : spin_chain.resonances)
					{
						// 生成中间态和共振态的legend
						std::vector<std::string> particles_for_legend;
						particles_for_legend.push_back(res_chain.intermediate);
						particles_for_legend.push_back(resonance);

						std::string legend = generateLegend(particles_for_legend);
						legends.push_back(legend);
					}
				}
			}
		}

		return legends;
	}

	// 获取所有可能的共振态组合的legend
	std::vector<std::string> getAllResonanceCombinationLegends() const
	{
		std::vector<std::string> legends;

		for (const auto &chain : decay_chains_)
		{
			// 构建中间态到共振态的映射
			std::map<std::pair<std::string, std::vector<int>>, std::vector<std::string>> spin_resonance_map;
			std::vector<std::vector<Particle>> particleLists;

			for (const auto &res_chain : chain.resonance_chains)
			{
				std::vector<Particle> particles;
				for (const auto &spin_chain : res_chain.spin_chains)
				{
					std::pair<std::string, std::vector<int>> key = {res_chain.intermediate,
																	{spin_chain.spin_parity[0], spin_chain.spin_parity[1]}};
					spin_resonance_map[key] = spin_chain.resonances;
					particles.push_back({res_chain.intermediate, spin_chain.spin_parity[0],
										 spin_chain.spin_parity[1], -1.0});
				}
				particleLists.push_back(particles);
			}

			// 生成所有JP组合
			std::vector<std::vector<Particle>> result = {{}};
			for (const auto &particleList : particleLists)
			{
				std::vector<std::vector<Particle>> temp;
				for (const auto &res : result)
				{
					for (const auto &particle : particleList)
					{
						std::vector<Particle> new_res = res;
						new_res.push_back(particle);
						temp.push_back(new_res);
					}
				}
				result = std::move(temp);
			}

			// 为每个JP组合生成共振态组合的legend
			for (size_t i = 0; i < result.size(); ++i)
			{
				const auto &jp_combination = result[i];

				// 为当前JP组合生成所有共振态组合
				std::vector<std::vector<std::pair<std::string, std::string>>> resonance_combinations = {{}};

				for (const auto &particle : jp_combination)
				{
					std::pair<std::string, std::vector<int>> key = {particle.name, {particle.spin, particle.parity}};
					const auto &resonances = spin_resonance_map[key];

					std::vector<std::vector<std::pair<std::string, std::string>>> temp_res;
					for (const auto &current_combo : resonance_combinations)
					{
						for (const auto &resonance : resonances)
						{
							std::vector<std::pair<std::string, std::string>> new_combo = current_combo;
							new_combo.push_back({particle.name, resonance});
							temp_res.push_back(new_combo);
						}
					}
					resonance_combinations = std::move(temp_res);
				}

				// 为每个共振态组合生成legend
				for (const auto &res_combo : resonance_combinations)
				{
					std::vector<std::string> particles_for_legend;
					for (const auto &res_pair : res_combo)
					{
						particles_for_legend.push_back(res_pair.second);
					}
					std::string legend = generateLegend(particles_for_legend);
					legends.push_back(legend);
				}
			}
		}

		return legends;
	}

private:
	// 生成定制legend：根据legend模板生成所有可能的组合
	std::vector<std::string> generateCustomLegends() const
	{
		std::vector<std::string> legends;

		for (const auto &chain : decay_chains_)
		{
			if (!chain.legend_template.empty())
			{
				// 检查legend模板中的占位符
				std::map<std::string, std::vector<std::string>> placeholder_map;
				std::vector<std::string> template_items;

				for (const auto &item : chain.legend_template)
				{
					// 检查是否是中间态占位符（以R_开头）
					if (item.find("R_") == 0)
					{
						// 如果这个占位符还没有处理过，找到对应的共振态链
						if (placeholder_map.find(item) == placeholder_map.end())
						{
							std::vector<std::string> resonances;
							for (const auto &res_chain : chain.resonance_chains)
							{
								if (res_chain.intermediate == item)
								{
									for (const auto &spin_chain : res_chain.spin_chains)
									{
										resonances.insert(resonances.end(),
														  spin_chain.resonances.begin(),
														  spin_chain.resonances.end());
									}
									break;
								}
							}
							placeholder_map[item] = resonances;
						}
					}
					template_items.push_back(item);
				}

				// 生成所有组合
				if (!placeholder_map.empty())
				{
					// 获取唯一的占位符列表
					std::vector<std::string> unique_placeholders;
					for (const auto &[placeholder, resonances] : placeholder_map)
					{
						unique_placeholders.push_back(placeholder);
					}

					// 为每个唯一占位符生成共振态组合
					std::vector<std::vector<std::string>> combinations = {{}};
					for (const auto &placeholder : unique_placeholders)
					{
						const auto &resonance_list = placeholder_map[placeholder];
						std::vector<std::vector<std::string>> temp;
						for (const auto &current_combo : combinations)
						{
							for (const auto &resonance : resonance_list)
							{
								std::vector<std::string> new_combo = current_combo;
								new_combo.push_back(resonance);
								temp.push_back(new_combo);
							}
						}
						combinations = std::move(temp);
					}

					// 为每个组合生成legend
					for (const auto &combo : combinations)
					{
						std::vector<std::string> particles_for_legend;

						// 创建占位符到共振态的映射
						std::map<std::string, std::string> placeholder_to_resonance;
						for (size_t i = 0; i < unique_placeholders.size(); ++i)
						{
							placeholder_to_resonance[unique_placeholders[i]] = combo[i];
						}

						// 构建legend
						for (const auto &item : template_items)
						{
							if (item.find("R_") == 0)
							{
								particles_for_legend.push_back(placeholder_to_resonance[item]);
							}
							else
							{
								particles_for_legend.push_back(item);
							}
						}

						std::string legend = generateLegend(particles_for_legend);
						legends.push_back(legend);
					}
				}
				else
				{
					// 如果没有占位符，直接生成legend
					std::string legend = generateLegend(chain.legend_template);
					legends.push_back(legend);
				}
			}
		}

		return legends;
	}

	// 获取每个decay chain的定制legend
	std::map<std::string, std::vector<std::string>> generateChainCustomLegends() const
	{
		std::map<std::string, std::vector<std::string>> chain_legends;

		for (const auto &chain : decay_chains_)
		{
			if (!chain.legend_template.empty())
			{
				std::vector<std::string> legends;

				// 检查legend模板中的占位符
				std::map<std::string, std::vector<std::string>> placeholder_map;
				std::vector<std::string> template_items;

				for (const auto &item : chain.legend_template)
				{
					// 检查是否是中间态占位符（以R_开头）
					if (item.find("R_") == 0)
					{
						// 如果这个占位符还没有处理过，找到对应的共振态链
						if (placeholder_map.find(item) == placeholder_map.end())
						{
							std::vector<std::string> resonances;
							for (const auto &res_chain : chain.resonance_chains)
							{
								if (res_chain.intermediate == item)
								{
									for (const auto &spin_chain : res_chain.spin_chains)
									{
										resonances.insert(resonances.end(),
														  spin_chain.resonances.begin(),
														  spin_chain.resonances.end());
									}
									break;
								}
							}
							placeholder_map[item] = resonances;
						}
					}
					template_items.push_back(item);
				}

				// 生成所有组合
				if (!placeholder_map.empty())
				{
					// 获取唯一的占位符列表
					std::vector<std::string> unique_placeholders;
					for (const auto &[placeholder, resonances] : placeholder_map)
					{
						unique_placeholders.push_back(placeholder);
					}

					// 为每个唯一占位符生成共振态组合
					std::vector<std::vector<std::string>> combinations = {{}};
					for (const auto &placeholder : unique_placeholders)
					{
						const auto &resonance_list = placeholder_map[placeholder];
						std::vector<std::vector<std::string>> temp;
						for (const auto &current_combo : combinations)
						{
							for (const auto &resonance : resonance_list)
							{
								std::vector<std::string> new_combo = current_combo;
								new_combo.push_back(resonance);
								temp.push_back(new_combo);
							}
						}
						combinations = std::move(temp);
					}

					// 为每个组合生成legend
					for (const auto &combo : combinations)
					{
						std::vector<std::string> particles_for_legend;

						// 创建占位符到共振态的映射
						std::map<std::string, std::string> placeholder_to_resonance;
						for (size_t i = 0; i < unique_placeholders.size(); ++i)
						{
							placeholder_to_resonance[unique_placeholders[i]] = combo[i];
						}

						// 构建legend
						for (const auto &item : template_items)
						{
							if (item.find("R_") == 0)
							{
								particles_for_legend.push_back(placeholder_to_resonance[item]);
							}
							else
							{
								particles_for_legend.push_back(item);
							}
						}

						std::string legend = generateLegend(particles_for_legend);
						legends.push_back(legend);
					}
				}
				else
				{
					// 如果没有占位符，直接生成legend
					std::string legend = generateLegend(chain.legend_template);
					legends.push_back(legend);
				}

				chain_legends[chain.name] = legends;
			}
		}

		return chain_legends;
	}

	void parseParticles(const YAML::Node &node)
	{
		for (const auto &particle_node : node)
		{
			std::string name = particle_node.first.as<std::string>();
			const auto &props = particle_node.second;

			Particle particle;
			particle.name = name;
			particle.spin = props["J"].as<int>();
			particle.parity = props["P"].as<int>();
			particle.mass = props["mass"].as<double>();
			if (props["tex"])
			{
				particle.tex = props["tex"].as<std::string>();
			}
			else
			{
				particle.tex = name; // 如果没有tex字段，使用name作为默认值
			}

			particles_.push_back(particle);
		}
	};

	void parseData(const YAML::Node &node)
	{
		if (node["dat_order"])
		{
			dat_order_ = node["dat_order"].as<std::vector<std::string>>();
		}

		if (node["data"])
		{
			data_files_["data"] = node["data"].as<std::vector<std::string>>();
		}

		if (node["phsp"])
		{
			data_files_["phsp"] = node["phsp"].as<std::vector<std::string>>();
		}

		if (node["bkg"])
		{
			data_files_["bkg"] = node["bkg"].as<std::vector<std::string>>();
		}
	};

	void parseDecayChains(const YAML::Node &node)
	{
		for (const auto &chain_node : node)
		{
			std::string chain_name = chain_node.first.as<std::string>();

			DecayChainConfig chain;
			chain.name = chain_name;

			// 解析衰变步骤
			if (chain_node.second["decay"])
			{
				const auto &decay_steps = chain_node.second["decay"];
				for (const auto &step_node : decay_steps)
				{
					// 解析 {mother, [daughter1, daughter2, ...]} 格式
					if (step_node.IsMap())
					{
						for (const auto &decay_pair : step_node)
						{
							DecayStep step;
							step.mother = decay_pair.first.as<std::string>();
							step.daughters = decay_pair.second.as<std::vector<std::string>>();
							chain.decay_steps.push_back(step);
						}
					}
				}
			}

			// 解析共振态链配置
			for (const auto &res_chain_node : chain_node.second)
			{
				std::string key = res_chain_node.first.as<std::string>();
				if (key != "decay" && key != "legend")
				{
					ResonanceChainConfig res_chain;
					res_chain.intermediate = key;

					for (const auto &spin_config : res_chain_node.second)
					{
						// Parse new format: [{J: 1}, {P: -1}]: [resonances...]
						// spin_config is a sequence of maps
						for (const auto &spin_pair : spin_config)
						{
							SpinChainConfig spin_chain;

							// The key is actually a sequence: [{J: 1}, {P: -1}]
							// We need to extract J and P from this sequence
							std::vector<int> spin_parity;

							if (spin_pair.first.IsSequence())
							{
								for (const auto &j_p_node : spin_pair.first)
								{
									if (j_p_node.IsMap())
									{
										if (j_p_node["J"])
										{
											spin_parity.push_back(j_p_node["J"].as<int>());
										}
										if (j_p_node["P"])
										{
											spin_parity.push_back(j_p_node["P"].as<int>());
										}
									}
								}
							}
							else
							{
								// Fallback: try to parse as vector (old format)
								spin_parity = spin_pair.first.as<std::vector<int>>();
							}

							spin_chain.spin_parity = spin_parity;
							spin_chain.resonances = spin_pair.second.as<std::vector<std::string>>();
							res_chain.spin_chains.push_back(spin_chain);
						}
					}

					chain.resonance_chains.push_back(res_chain);
				}
			}

			// 解析legend模板
			if (chain_node.second["legend"])
			{
				chain.legend_template = chain_node.second["legend"].as<std::vector<std::string>>();
			}

			decay_chains_.push_back(chain);
		}
	};

	void parseResonances(const YAML::Node &node)
	{
		for (const auto &res_node : node)
		{
			std::string name = res_node.first.as<std::string>();
			const auto &props = res_node.second;

			ResonanceConfig res;
			res.name = name;
			if (props["J"])
			{
				res.J = props["J"].as<int>();
			}
			else
			{
				res.J = -1; // 如果没有J字段，设为-1
			}
			if (props["P"])
			{
				res.P = props["P"].as<int>();
			}
			else
			{
				res.P = 0; // 如果没有P字段，设为0
			}
			res.type = props["model"].as<std::string>();
			res.parameters = props["parameters"].as<std::vector<double>>();
			if (props["tex"])
			{
				res.tex = props["tex"].as<std::string>();
			}
			else
			{
				res.tex = name; // 如果没有tex字段，使用name作为默认值
			}

			resonances_[name] = res;
		}
	};

	void parseConjugatePairs(const YAML::Node &node)
	{
		for (const auto &pair_node : node)
		{
			auto pair = pair_node.as<std::vector<std::string>>();
			if (pair.size() == 2)
			{
				conjugate_pairs_.emplace_back(pair[0], pair[1]);
			}
		}
	};

	std::vector<Particle> particles_;
	std::vector<DecayChainConfig> decay_chains_;
	std::map<std::string, ResonanceConfig> resonances_;
	std::vector<std::pair<std::string, std::string>> conjugate_pairs_;
	std::map<std::string, std::vector<std::string>> data_files_;
	std::vector<std::string> dat_order_;
};

class AmplitudeCalculator
{
private:
	ConfigParser config_parser_;
	std::vector<Particle> particles_;
	std::unordered_map<std::string, Resonance> resonances_;
	std::vector<std::pair<std::string, std::string>> conjugate_pairs_;

	// 振幅计算相关
	int n_amplitudes_ = 0;
	std::vector<std::string> amplitude_names_;
	// std::unordered_map<std::string, int> name_to_index_;
	std::vector<std::pair<int, int>> conjugate_pair_indices_;
	int nPolar_ = 1;
	std::vector<int> nSLvectors_;

public:
	AmplitudeCalculator(const std::string &config_file) : config_parser_(config_file)
	{
		std::cout << "AmplitudeCalculator: Config parsed successfully" << std::endl;
		std::cout << "  Particles: " << config_parser_.getParticles().size() << std::endl;
		std::cout << "  Decay chains: " << config_parser_.getDecayChains().size() << std::endl;
		std::cout << "  Resonances: " << config_parser_.getResonances().size() << std::endl;
		std::cout << "  Conjugate pairs: " << config_parser_.getConjugatePairs().size() << std::endl;

		// auto legends = config_parser_.getCustomLegends();
		// for (const auto &leg : legends)
		// {
		// 	std::cout << leg << std::endl;
		// }

		auto legends = config_parser_.getCustomLegends();
		std::ofstream file("legend.txt");
		if (file.is_open())
		{
			for (const auto &leg : legends)
			{
				file << leg << std::endl;
			}
			file.close();
			// std::cout << "Legends written to legend.txt" << std::endl;
		}
		else
		{
			std::cerr << "Unable to open legend.txt for writing" << std::endl;
		}

		initializeFromConfig();
	}

private:
	void initializeFromConfig()
	{
		// 1. 初始化粒子
		initializeParticles();

		// 2. 初始化共振态
		initializeResonances();

		// 3. 设置共轭对
		initializeConjugatePairs();
	}

	void initializeParticles()
	{
		const auto &config_particles = config_parser_.getParticles();
		for (const auto &particle_config : config_particles)
		{
			particles_.push_back(particle_config);
		}
	}

	void initializeResonances()
	{
		const auto &config_resonances = config_parser_.getResonances();
		const auto &decay_chains = config_parser_.getDecayChains();

		// 首先创建所有共振态对象
		for (const auto &[name, res_config] : config_resonances)
		{
			auto spin_parity = findSpinParityFromChains(name, decay_chains);
			auto intermediate = findIntermediateFromChains(name, decay_chains);

			if ((spin_parity.first == 0 && spin_parity.second == 0) || (intermediate == "unknown"))
			{
				continue;
			}

			if (res_config.J == spin_parity.first && res_config.P == spin_parity.second)
			{
				resonances_.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(name, intermediate, spin_parity.first, spin_parity.second, res_config.type, res_config.parameters));
				// std::cout << "Resonance " << name << " spin/parity from config matches chains: [" << res_config.J << ", " << res_config.P << "]" << std::endl;
			}
			// else
			// {
			// 	std::cout << "Warning: Resonance " << name << " spin/parity from config ("
			// 			  << res_config.J << ", " << res_config.P << ") does not match chains ("
			// 			  << spin_parity.first << ", " << spin_parity.second << "). Using values from chains." << std::endl;
			// }
		}
	}

	void initializeConjugatePairs()
	{
		conjugate_pairs_ = config_parser_.getConjugatePairs();

		// 验证共轭对是否存在
		for (const auto &pair : conjugate_pairs_)
		{
			auto it1 = resonances_.find(pair.first);
			auto it2 = resonances_.find(pair.second);

			if (it1 == resonances_.end() || it2 == resonances_.end())
			{
				std::cerr << "Warning: Conjugate pair " << pair.first
						  << " <-> " << pair.second << " not found in resonances" << std::endl;
			}
			else
			{
				// std::cout << pair.first << " <-> " << pair.second << std::endl;
				it1->second.setConjugatePartner(pair.second);
				it2->second.setConjugatePartner(pair.first);
			}
		}
	}

	std::pair<int, int> findSpinParityFromChains(const std::string &res_name, const std::vector<DecayChainConfig> &decay_chains)
	{
		for (const auto &chain : decay_chains)
		{
			for (const auto &res_chain : chain.resonance_chains)
			{
				for (const auto &spin_chain : res_chain.spin_chains)
				{

					for (const auto &res : spin_chain.resonances)
					{
						if (res == res_name && spin_chain.spin_parity.size() >= 2)
						{
							// std::cout << res_chain.intermediate << " resonance: " << res_name << " Spin/Parity: [" << spin_chain.spin_parity[0] << ", " << spin_chain.spin_parity[1] << "]" << std::endl;
							return {spin_chain.spin_parity[0], spin_chain.spin_parity[1]};
						}
					}
				}
			}
		}

		// std::cerr << "Warning: Could not find spin/parity for resonance " << res_name
		// 		  << ", using default values (1, -1)" << std::endl;
		return {0, 0};
	}

	std::string findIntermediateFromChains(const std::string &res_name,
										   const std::vector<DecayChainConfig> &decay_chains)
	{
		for (const auto &chain : decay_chains)
		{
			for (const auto &res_chain : chain.resonance_chains)
			{
				for (const auto &spin_chain : res_chain.spin_chains)
				{
					for (const auto &res : spin_chain.resonances)
					{
						if (res == res_name)
						{
							// std::cout << "resonance: " << res_name << " intermediat: " << res_chain.intermediate << std::endl;
							return res_chain.intermediate;
						}
					}
				}
			}
		}

		// std::cerr << "Warning: Could not find intermediate for resonance " << res_name
		// 		  << ", using default" << std::endl;
		return "unknown";
	}

public:
	// 主要的振幅计算方法
	cuDoubleComplex *calculateAmplitudes(
		const std::map<std::string, std::vector<LorentzVector>> &Vp4,
		const int print_flag)
	{
		// std::vector<cuDoubleComplex *> all_amplitudes;
		resetAmplitudeTracking();
		// std::vector<int> nSLvectors;

		// auto chains = getSelectedChains(selected_chains);
		auto chains = config_parser_.getDecayChains();

		// 收集所有共振态名称用于建立共轭关系
		std::vector<std::string> all_resonance_names;
		all_resonance_names = collectAllResonanceNames(chains);
		setupConjugatePairIndices(all_resonance_names);

		size_t nGls = 0;

		// 获取总振幅长度
		for (const auto &chain : chains)
		{
			// std::cout << "Chain name: " << chain.name << std::endl;

			std::map<std::pair<std::string, std::vector<int>>, std::vector<std::string>> spin_resonance_map;
			std::vector<std::vector<Particle>> resLists;
			for (const auto &res_chain : chain.resonance_chains)
			{
				std::vector<Particle> particles;
				for (const auto &spin_chain : res_chain.spin_chains)
				{
					std::pair<std::string, std::vector<int>> key = {res_chain.intermediate, {spin_chain.spin_parity[0], spin_chain.spin_parity[1]}}; //
					spin_resonance_map[key] = spin_chain.resonances;
					particles.push_back({res_chain.intermediate, static_cast<int>(spin_chain.spin_parity[0]), static_cast<int>(spin_chain.spin_parity[1]), -1});
				}
				resLists.push_back(particles);
			}

			// 生成并输出所有组合
			std::vector<std::vector<Particle>> jpcombs = {{}};
			for (const auto &particleList : resLists)
			{
				std::vector<std::vector<Particle>> temp;
				for (const auto &jpcomb : jpcombs)
				{
					for (const auto &particle : particleList)
					{
						std::vector<Particle> new_res = jpcomb;
						new_res.push_back(particle);
						temp.push_back(new_res);
					}
				}
				jpcombs = std::move(temp);
			}

			for (auto jpcomb : jpcombs)
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

					for (auto res_jp : jpcomb)
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

					if (print_flag == 1)
					{
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
				}
				// std::cout << std::endl;

				auto slcombs = cas.getSLCombinations();
				int nPolar = cas.computeNPolarizations(Vp4);
				nPolar_ = nPolar;

				if (print_flag == 1)
				{
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
				}

				std::vector<std::vector<std::pair<std::string, std::string>>> resonance_combinations = {{}};
				for (const auto particle : jpcomb)
				{
					std::pair<std::string, std::vector<int>> key = {particle.name, {particle.spin, particle.parity}};
					const auto &resonances = spin_resonance_map[key];

					// 扩展共振态组合
					std::vector<std::vector<std::pair<std::string, std::string>>> temp_res;
					for (const auto &current_combo : resonance_combinations)
					{
						for (const auto &resonance : resonances)
						{
							std::vector<std::pair<std::string, std::string>> new_combo = current_combo;
							new_combo.push_back({particle.name, resonance});
							temp_res.push_back(new_combo);
							nGls += slcombs.size();
						}
					}
					resonance_combinations = std::move(temp_res);
				}

				if (print_flag == 1)
				{
					std::cout << "Resonance: ";
					for (size_t k = 0; k < resonance_combinations.size(); ++k)
					{
						std::cout << "{ ";
						for (size_t j = 0; j < resonance_combinations[k].size(); ++j)
						{
							const auto &res_pair = resonance_combinations[k][j];
							std::cout << res_pair.second; // 共振态名称
							if (j < resonance_combinations[k].size() - 1)
								std::cout << ", ";
						}
						if (k < resonance_combinations.size() - 1)
							std::cout << " }, ";
						else
							std::cout << "}";
					}
					std::cout << std::endl;
				}
			}
		}

		int n_events = Vp4.begin()->second.size();
		// int amplitude_size = n_events * nPolar_;
		cuDoubleComplex *d_all_amplitudes;
		const size_t total_amplitudes = nGls * n_events * nPolar_;
		cudaMalloc(&d_all_amplitudes, total_amplitudes * sizeof(cuDoubleComplex));

		// std::cout << "Total amplitudes (nGls): " << nGls << std::endl;
		// std::cout << n_events << " " << nPolar_ << std::endl;

		// 计算总振幅
		int gls_index = 0;
		for (const auto &chain : chains)
		{
			// std::cout << "Chain name: " << chain.name << std::endl;

			std::map<std::pair<std::string, std::vector<int>>, std::vector<std::string>> spin_resonance_map;
			std::vector<std::vector<Particle>> resLists;
			for (const auto &res_chain : chain.resonance_chains)
			{
				std::vector<Particle> particles;
				for (const auto &spin_chain : res_chain.spin_chains)
				{
					std::pair<std::string, std::vector<int>> key = {res_chain.intermediate, {spin_chain.spin_parity[0], spin_chain.spin_parity[1]}}; //
					spin_resonance_map[key] = spin_chain.resonances;
					particles.push_back({res_chain.intermediate, static_cast<int>(spin_chain.spin_parity[0]), static_cast<int>(spin_chain.spin_parity[1]), -1});
				}
				resLists.push_back(particles);
			}

			// 生成并输出所有组合
			std::vector<std::vector<Particle>> jpcombs = {{}};
			for (const auto &particleList : resLists)
			{
				std::vector<std::vector<Particle>> temp;
				for (const auto &jpcomb : jpcombs)
				{
					for (const auto &particle : particleList)
					{
						std::vector<Particle> new_res = jpcomb;
						new_res.push_back(particle);
						temp.push_back(new_res);
					}
				}
				jpcombs = std::move(temp);
			}

			for (auto jpcomb : jpcombs)
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

					for (auto res_jp : jpcomb)
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
				}
				// std::cout << std::endl;

				auto slcombs = cas.getSLCombinations();

				std::vector<std::vector<std::pair<std::string, std::string>>> resonance_combinations = {{}};
				for (const auto particle : jpcomb)
				{
					std::pair<std::string, std::vector<int>> key = {particle.name, {particle.spin, particle.parity}};
					const auto &resonances = spin_resonance_map[key];

					// 扩展共振态组合
					std::vector<std::vector<std::pair<std::string, std::string>>> temp_res;
					for (const auto &current_combo : resonance_combinations)
					{
						for (const auto &resonance : resonances)
						{
							std::vector<std::pair<std::string, std::string>> new_combo = current_combo;
							new_combo.push_back({particle.name, resonance});
							temp_res.push_back(new_combo);
						}
					}
					resonance_combinations = std::move(temp_res);
				}

				cas.computeSLAmps(Vp4);
				int nSLcombs = cas.getNSLCombs();
				int nEvents = cas.getNEvents();

				for (const auto resonance : resonance_combinations)
				{
					std::vector<Resonance> resforAmp;
					std::string full_name = chain.name;
					for (const auto &res : resonance)
					{
						auto it = resonances_.find(res.second);
						if (it != resonances_.end())
						{
							full_name += "-" + res.first + "-" + res.second;
							resforAmp.push_back(it->second);
						}
						else
						{
							std::cerr << "Warning: Resonance " << res << " not found in chain " << chain.name << std::endl;
						}
						// resforAmp.push_back(it->second);
					}

					// cuDoubleComplex *amp = cas.getAmps(resforAmp);
					// all_amplitudes.push_back(amp);
					cas.getAmps(d_all_amplitudes, resforAmp, gls_index);
					nSLvectors_.push_back(nSLcombs);
					gls_index += nSLcombs;

					// 记录振幅信息
					amplitude_names_.push_back(full_name);
					// name_to_index_[resName] = n_amplitudes_;
					n_amplitudes_ += nSLcombs;
				}
			}
		}

		// for (size_t i = 0; i < nSLvectors_.size(); ++i)
		// {
		// 	std::cout << "nSLvectors[" << i << "] = " << nSLvectors_[i] << std::endl;
		// }

		// std::cout << __LINE__ << std::endl;

		// 合并所有振幅到设备内存
		// cuDoubleComplex *device_result = combineAmplitudesDevice(all_amplitudes, amplitude_size, nSLvectors_);

		// 清理临时设备内存
		// cleanupTemporaryAmplitudes(all_amplitudes);

		return d_all_amplitudes;
	}

private:
	void resetAmplitudeTracking()
	{
		n_amplitudes_ = 0;
		amplitude_names_.clear();
		conjugate_pair_indices_.clear();
		nSLvectors_.clear();
	}

	std::vector<std::string> collectAllResonanceNames(const std::vector<DecayChainConfig> &chains)
	{
		std::vector<std::string> all_names;

		for (const auto &chain : chains)
		{
			for (const auto &res_chain : chain.resonance_chains)
			{
				for (const auto &spin_chain : res_chain.spin_chains)
				{
					for (const auto &res_name : spin_chain.resonances)
					{
						// std::cout << "resonance: " << res_name << std::endl;
						all_names.push_back(res_name);
					}
				}
			}
		}

		return all_names;
	}

	void setupConjugatePairIndices(const std::vector<std::string> &all_resonance_names)
	{
		std::unordered_map<std::string, int> temp_indices;
		for (int i = 0; i < all_resonance_names.size(); ++i)
		{
			temp_indices[all_resonance_names[i]] = i;
		}

		for (const auto &pair : conjugate_pairs_)
		{
			if (temp_indices.count(pair.first) && temp_indices.count(pair.second))
			{
				int idx1 = temp_indices[pair.first];
				int idx2 = temp_indices[pair.second];

				if (idx1 < idx2)
				{
					conjugate_pair_indices_.emplace_back(idx1, idx2);
				}
				else
				{
					conjugate_pair_indices_.emplace_back(idx2, idx1);
				}
			}
		}
	}

	cuDoubleComplex *combineAmplitudesDevice(const std::vector<cuDoubleComplex *> &amplitudes, int amplitude_size, const std::vector<int> &nSLvectors)
	{
		if (amplitudes.empty() || amplitudes.size() != nSLvectors.size())
		{
			return nullptr;
		}

		// 计算总大小
		size_t total_size = 0;
		for (size_t i = 0; i < nSLvectors.size(); ++i)
		{
			total_size += nSLvectors[i] * amplitude_size;
		}

		cuDoubleComplex *combined_device = nullptr;

		cudaError_t cudaStatus = cudaMalloc(&combined_device, total_size * sizeof(cuDoubleComplex));
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "cudaMalloc failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
			return nullptr;
		}

		// 逐个拷贝振幅数据
		size_t current_offset = 0;
		for (size_t i = 0; i < amplitudes.size(); ++i)
		{
			if (amplitudes[i] == nullptr)
			{
				std::cerr << "Warning: amplitudes[" << i << "] is null!" << std::endl;
				// 跳过这个振幅，但需要更新偏移量
				current_offset += nSLvectors[i] * amplitude_size;
				continue;
			}

			size_t current_size = nSLvectors[i] * amplitude_size;

			cudaStatus = cudaMemcpy(combined_device + current_offset,
									amplitudes[i],
									current_size * sizeof(cuDoubleComplex),
									cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess)
			{
				std::cerr << "cudaMemcpy failed for amplitudes[" << i << "]! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
				cudaFree(combined_device);
				return nullptr;
			}

			current_offset += current_size;
		}

		return combined_device;
	}

	void cleanupTemporaryAmplitudes(std::vector<cuDoubleComplex *> &amplitudes)
	{
		for (auto ptr : amplitudes)
		{
			if (ptr != nullptr)
			{
				cudaFree(ptr);
			}
		}
		amplitudes.clear();
	}

public:
	// 公共接口方法
	const std::vector<std::string> &getAmplitudeNames() const { return amplitude_names_; }
	// const std::unordered_map<std::string, int> &getNameToIndexMap() const { return name_to_index_; }
	const std::vector<std::pair<int, int>> &getConjugatePairIndices() const { return conjugate_pair_indices_; }
	int getNAmplitudes() const { return n_amplitudes_; }
	int getNPolarization() const { return nPolar_; }
	const std::vector<int> getNSLVectors() const { return nSLvectors_; }

	// 获取配置信息
	const ConfigParser &getConfigParser() const { return config_parser_; }
	const std::vector<Particle> &getParticles() const { return particles_; }
	const std::unordered_map<std::string, Resonance> &getResonances() const { return resonances_; }
};

class NLLFunction : public torch::autograd::Function<NLLFunction>
{
public:
	static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor &vector, int n_gls_, int n_polar_, const cuDoubleComplex *data_fix_, int data_length, const cuDoubleComplex *phsp_fix_, int phsp_length, const cuDoubleComplex *bkg_fix_, int bkg_length, std::vector<std::pair<int, int>> conjugate_pairs_)
	{
		TORCH_CHECK(vector.is_cuda(), "[NLLForward] vector must be on CUDA");
		TORCH_CHECK(vector.dtype() == c10::kComplexDouble, "[NLLForward] vector must be complex128");

		// 获取当前设备并设置
		const int target_dev = vector.get_device();
		torch::Device dev(torch::kCUDA, target_dev);

		// 延长vector以处理共轭对
		torch::Tensor extended_vector = extendVectorWithConjugates(vector, conjugate_pairs_, dev);
		const int extended_n_gls = extended_vector.numel();

		// 后续逻辑（MC因子计算等）
		cuDoubleComplex *d_B = nullptr;
		double *d_mc_amp = nullptr;
		cudaMalloc(&d_B, phsp_length * sizeof(cuDoubleComplex));
		cudaMalloc(&d_mc_amp, sizeof(double));

		// computeSingleResult(phsp_fix_, reinterpret_cast<const cuDoubleComplex *>(extended_vector.data_ptr()), d_B, d_mc_amp, phsp_length, extended_n_gls);
		computePHSPfactor(phsp_fix_, reinterpret_cast<const cuDoubleComplex *>(extended_vector.data_ptr()), d_B, d_mc_amp, phsp_length, extended_n_gls);

		double h_phsp_factor;
		cudaMemcpy(&h_phsp_factor, d_mc_amp, sizeof(double), cudaMemcpyDeviceToHost);
		h_phsp_factor = h_phsp_factor / static_cast<double>(phsp_length / n_polar_);

		// NLL计算
		cuDoubleComplex *d_S = nullptr;
		cuDoubleComplex *d_Q = nullptr;
		double *d_data_nll = nullptr;
		const int Q_numel = data_length / n_polar_;
		cudaMalloc(&d_S, data_length * sizeof(cuDoubleComplex));
		cudaMalloc(&d_Q, Q_numel * sizeof(cuDoubleComplex));
		cudaMalloc(&d_data_nll, sizeof(double));

		computeNll(data_fix_, reinterpret_cast<const cuDoubleComplex *>(extended_vector.data_ptr()), d_S, d_Q, d_data_nll, data_length, extended_n_gls, n_polar_, h_phsp_factor);

		double h_data_nll;
		cudaMemcpy(&h_data_nll, d_data_nll, sizeof(double), cudaMemcpyDeviceToHost);

		// bkg部分
		cuDoubleComplex *d_bkg_S = nullptr;
		cuDoubleComplex *d_bkg_Q = nullptr;
		double *d_bkg_nll = nullptr;
		const int bkg_Q_numel = bkg_length / n_polar_;
		cudaMalloc(&d_bkg_S, bkg_length * sizeof(cuDoubleComplex));
		cudaMalloc(&d_bkg_Q, bkg_Q_numel * sizeof(cuDoubleComplex));
		cudaMalloc(&d_bkg_nll, sizeof(double));
		double h_bkg_nll = 0.0;
		if (bkg_fix_ != nullptr && bkg_length > 0)
		{
			computeNll(bkg_fix_, reinterpret_cast<const cuDoubleComplex *>(extended_vector.data_ptr()), d_bkg_S, d_bkg_Q, d_bkg_nll, bkg_length, extended_n_gls, n_polar_, h_phsp_factor);

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
		cuDoubleComplex *d_B = reinterpret_cast<cuDoubleComplex *>(ctx->saved_data["d_B_ptr"].toInt());
		cuDoubleComplex *data_fix = reinterpret_cast<cuDoubleComplex *>(ctx->saved_data["data_fix_ptr"].toInt());
		cuDoubleComplex *phsp_fix = reinterpret_cast<cuDoubleComplex *>(ctx->saved_data["phsp_fix_ptr"].toInt());
		cuDoubleComplex *d_S = reinterpret_cast<cuDoubleComplex *>(ctx->saved_data["d_S_ptr"].toInt());
		cuDoubleComplex *d_Q = reinterpret_cast<cuDoubleComplex *>(ctx->saved_data["d_Q_ptr"].toInt());

		// // 获取背景数据的指针（如果存在）
		// if (bkg_length > 0)
		// {
		cuDoubleComplex *bkg_fix = reinterpret_cast<cuDoubleComplex *>(ctx->saved_data["bkg_fix_ptr"].toInt());
		cuDoubleComplex *d_bkg_S = reinterpret_cast<cuDoubleComplex *>(ctx->saved_data["d_bkg_S_ptr"].toInt());
		cuDoubleComplex *d_bkg_Q = reinterpret_cast<cuDoubleComplex *>(ctx->saved_data["d_bkg_Q_ptr"].toInt());
		// }

		// 获取保存的变量
		const auto saved = ctx->get_saved_variables();
		const auto &original_vector = saved[0];
		const auto &extended_vector = saved[1];

		// 计算扩展向量的梯度
		cuDoubleComplex *d_extended_grad = nullptr;
		cudaMalloc(&d_extended_grad, extended_n_gls * sizeof(cuDoubleComplex));

		cublasHandle_t cublas_handle;
		cublasCreate(&cublas_handle);
		compute_gradient(data_fix, phsp_fix, d_S, d_Q, d_B, h_phsp_factor, extended_n_gls, data_length / n_polar, n_polar, phsp_length, d_extended_grad, cublas_handle);

		// 如果有背景数据，减去背景NLL的梯度
		if (bkg_fix != nullptr && bkg_length > 0)
		{
			cuDoubleComplex *d_bkg_extended_grad = nullptr;
			cudaMalloc(&d_bkg_extended_grad, extended_n_gls * sizeof(cuDoubleComplex));

			// 初始化背景梯度为0
			cudaMemset(d_bkg_extended_grad, 0, extended_n_gls * sizeof(cuDoubleComplex));

			// 计算背景NLL的梯度
			compute_gradient(bkg_fix, phsp_fix, d_bkg_S, d_bkg_Q, d_B, h_phsp_factor,
							 extended_n_gls, bkg_length / n_polar, n_polar,
							 phsp_length, d_bkg_extended_grad, cublas_handle);

			// 从数据梯度中减去背景梯度：∇L = ∇(data_nll) - ∇(bkg_nll)
			// 使用cublas的向量减法操作
			const cuDoubleComplex minus_one = make_cuDoubleComplex(-1.0, 0.0);
			cublasZaxpy(cublas_handle, extended_n_gls,
						&minus_one, d_bkg_extended_grad, 1,
						d_extended_grad, 1);

			cudaFree(d_bkg_extended_grad);
		}

		// 将扩展梯度的共轭部分合并回原始梯度
		torch::Tensor extended_grad = torch::empty({extended_n_gls}, torch::kComplexDouble).to(original_vector.device());
		cudaMemcpy(extended_grad.data_ptr(), d_extended_grad,
				   extended_n_gls * sizeof(cuDoubleComplex),
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
		torch::Tensor extended_vector = torch::zeros({extended_size}, torch::kComplexDouble).to(device);

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
		torch::Tensor grad_vector = torch::zeros({original_size}, torch::kComplexDouble).to(extended_grad.device());

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
	analysis() : n_gls_(0), n_polar_(0), data_fix_(nullptr), data_length(0), phsp_fix_(nullptr), phsp_length(0), bkg_fix_(nullptr), bkg_length(0)
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
		TORCH_CHECK(vector.dtype() == torch::kComplexDouble, "vector must be complex128");

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
		torch::Tensor extended_vector = torch::zeros({extended_size}, torch::kComplexDouble).to(dev);

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
		computeWeightResult(phsp_fix_, reinterpret_cast<const cuDoubleComplex *>(extended_vector.data_ptr()), d_final_result, d_total_integral, d_partial_result, d_partial_sum, d_nSLvectors, npartials, n_events, n_gls_, n_polar_);

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
		cudaMemcpy(host_array.data(), data_fix_, data_length * n_gls_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

		// std::cout << "debug " << host_array[0] << std::endl;

		torch::Tensor output = torch::empty({data_length * n_gls_}, dtype(torch::kComplexDouble));
		output.copy_(torch::from_blob(host_array.data(), {data_length * n_gls_}, torch::kComplexDouble));

		// cudaFree(host_array);

		return output;
	}

	torch::Tensor getPhspTensor() const
	{
		std::vector<std::complex<double>> host_array(phsp_length * n_gls_);
		cudaMemcpy(host_array.data(), phsp_fix_, phsp_length * n_gls_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

		// std::cout << "debug " << host_array[0] << std::endl;

		torch::Tensor output = torch::empty({phsp_length * n_gls_}, dtype(torch::kComplexDouble));
		output.copy_(torch::from_blob(host_array.data(), {phsp_length * n_gls_}, torch::kComplexDouble));

		// cudaFree(host_array);

		return output;
	}

	torch::Tensor getBkgTensor() const
	{
		std::vector<std::complex<double>> host_array(bkg_length);
		cudaMemcpy(host_array.data(), bkg_fix_, bkg_length * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

		// std::cout << "debug " << host_array[0] << std::endl;

		torch::Tensor output = torch::empty({bkg_length}, dtype(torch::kComplexDouble));
		output.copy_(torch::from_blob(host_array.data(), {bkg_length}, torch::kComplexDouble));

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
	int n_polar_;
	std::vector<int> nSLvectors_;
	cuDoubleComplex *data_fix_;
	int data_length;
	cuDoubleComplex *phsp_fix_;
	int phsp_length;
	cuDoubleComplex *bkg_fix_;
	int bkg_length;

	// 四动量数据
	std::map<std::string, std::vector<LorentzVector>> Vp4_data_;
	std::map<std::string, std::vector<LorentzVector>> Vp4_phsp_;
	std::map<std::string, std::vector<LorentzVector>> Vp4_bkg_;

	// 添加共轭信息
	std::vector<std::pair<int, int>> conjugate_pairs_;
	std::vector<std::string> amplitude_names_;

	void initialize(std::string config_file = "config.yml")
	{
		// 读取配置文件
		// std::string config_file = "config.yml";
		std::cout << "Reading config file: " << config_file << std::endl;

		// 创建振幅计算器，自动读取配置
		AmplitudeCalculator calculator(config_file);

		// 获取配置信息
		const auto &config_parser = calculator.getConfigParser();
		const auto &data_files = config_parser.getDataFiles();
		const auto &dat_order = config_parser.getDatOrder();

		// 读取相空间数据
		Vp4_phsp_ = readMomentaFromDat(data_files.at("phsp")[0], dat_order);

		// 计算相空间振幅
		// std::cout << "Calculating phase space amplitudes..." << std::endl;
		phsp_fix_ = calculator.calculateAmplitudes(Vp4_phsp_, 1);
		std::cout << "Reading phase space data..." << std::endl;
		std::cout << "Phase space events: " << Vp4_phsp_.begin()->second.size() << std::endl;
		n_gls_ = calculator.getNAmplitudes();
		n_polar_ = calculator.getNPolarization(); // 假设每个事件有3个极化状态
		phsp_length = Vp4_phsp_.begin()->second.size() * n_polar_;

		// 读取数据
		std::cout << "Reading data..." << std::endl;
		Vp4_data_ = readMomentaFromDat(data_files.at("data")[0], dat_order);
		std::cout << "Data events: " << Vp4_data_.begin()->second.size() << std::endl;

		// 计算数据振幅
		// std::cout << "Calculating data amplitudes..." << std::endl;
		data_fix_ = calculator.calculateAmplitudes(Vp4_data_, 0);
		data_length = Vp4_data_.begin()->second.size() * n_polar_;

		// 读取背景数据
		if (data_files.count("bkg") > 0)
		{
			std::cout << "Reading background data..." << std::endl;
			Vp4_bkg_ = readMomentaFromDat(data_files.at("bkg")[0], dat_order);
			std::cout << "Background events: " << Vp4_bkg_.begin()->second.size() << std::endl;
			// 计算背景振幅
			// std::cout << "Calculating background amplitudes..." << std::endl;
			bkg_fix_ = calculator.calculateAmplitudes(Vp4_bkg_, 0);
			bkg_length = Vp4_bkg_.begin()->second.size() * n_polar_;
		}

		// 获取振幅名称和共轭对信息
		amplitude_names_ = calculator.getAmplitudeNames();
		conjugate_pairs_ = calculator.getConjugatePairIndices();
		nSLvectors_ = calculator.getNSLVectors();

		std::cout << "Number of partial waves (n_gls_): " << n_gls_ << std::endl;
		std::cout << "Number of amplitude names: " << amplitude_names_.size() << std::endl;
		std::cout << "Number of conjugate pairs: " << conjugate_pairs_.size() << std::endl;
		std::cout << "Initialization complete." << std::endl;
	}
};

// 定义Python模块
PYBIND11_MODULE(mypwa, m)
{
	m.doc() = "mypwa";
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
