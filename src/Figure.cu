#include <Figure.cuh>
#include <helicity.cuh>

// 质量计算
struct MassCalculator
{
    const LorentzVector *momenta;
    const int *particle_indices;
    int n_particles;
    int total_particles;

    __host__ __device__ double operator()(int event_idx) const
    {
        LorentzVector total;
        for (int i = 0; i < n_particles; ++i)
        {
            int particle_idx = particle_indices[i];
            const LorentzVector &p = momenta[event_idx * total_particles + particle_idx];
            total = total + p;
        }
        return total.M();
    }
};

void CalculateMassHist(
    LorentzVector *device_momenta,
    const std::map<std::string, int> &particleToIndex,
    const std::vector<MassHistConfig> &histConfigs,
    double *weights,
    std::vector<TH1F *> &outputHistograms,
    int nEvents, int nParticles)
{
    // 1. 检查输入
    if (outputHistograms.size() != histConfigs.size())
    {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size() << ")" << std::endl;
        return;
    }

    if (nEvents == 0)
    {
        std::cerr << "Warning: No events to process!" << std::endl;
        return;
    }

    // 2. 复制权重到设备（如果有的话）
    thrust::device_vector<double> device_weights;
    if (weights != nullptr)
    {
        // thrust::host_vector<double> host_weights(weights, weights + nEvents);
        // device_weights = host_weights;
        device_weights = thrust::device_vector<double>(weights, weights + nEvents);
    }

    // 3. 对每个直方图配置进行处理
    for (size_t configIdx = 0; configIdx < histConfigs.size(); ++configIdx)
    {
        const auto &config = histConfigs[configIdx];
        TH1F *hist = outputHistograms[configIdx];

        if (!hist)
        {
            std::cerr << "Error: outputHistograms[" << configIdx << "] is null!" << std::endl;
            continue;
        }

        hist->Reset();

        // 3.1 获取该配置需要的粒子索引
        std::vector<int> particleIndices;
        for (const auto &particleName : config.particles)
        {
            auto it = particleToIndex.find(particleName);
            if (it == particleToIndex.end())
            {
                std::cerr << "Error: Particle '" << particleName
                          << "' not found in particleToIndex map!" << std::endl;
                particleIndices.clear();
                break;
            }
            particleIndices.push_back(it->second);
        }

        if (particleIndices.empty())
        {
            std::cerr << "Warning: Config " << configIdx << " has no valid particles!" << std::endl;
            continue;
        }

        // 检查range大小
        if (config.range.size() < 2)
        {
            std::cerr << "Error: Config " << configIdx << " range size < 2!" << std::endl;
            continue;
        }

        double min_bin = config.range[0];
        double max_bin = config.range[1];
        int n_bins = config.bins;
        double bin_width = (max_bin - min_bin) / n_bins;

        // 3.2 将粒子索引复制到设备
        thrust::device_vector<int> device_particle_indices = particleIndices;
        int nParticlesInConfig = particleIndices.size();

        // 3.3 为每个事件计算不变质量（使用Thrust transform）
        thrust::device_vector<double> device_masses(nEvents);

        // 获取设备原始指针
        int *d_indices_ptr = thrust::raw_pointer_cast(device_particle_indices.data());

        // 创建索引序列 [0, 1, 2, ..., nEvents-1]
        thrust::device_vector<int> event_indices(nEvents);
        thrust::sequence(event_indices.begin(), event_indices.end(), 0);

        // 计算所有事件的质量
        thrust::transform(
            event_indices.begin(),
            event_indices.end(),
            device_masses.begin(),
            MassCalculator{
                device_momenta,
                d_indices_ptr,
                nParticlesInConfig,
                nParticles});

        // 3.4 创建直方图bin边界
        thrust::device_vector<double> bin_edges(n_bins + 1);
        for (int i = 0; i <= n_bins; ++i)
        {
            bin_edges[i] = min_bin + i * bin_width;
        }

        // 3.5 将质量分配到bins中（使用权重）
        thrust::device_vector<double> bin_counts(n_bins, 0.0);

        if (weights != nullptr)
        {
            // 使用transform计算bin索引
            thrust::device_vector<int> bin_indices(nEvents);

            thrust::transform(
                device_masses.begin(),
                device_masses.end(),
                bin_indices.begin(),
                [=] __device__(double mass) -> int
                {
                    if (mass < min_bin || mass >= max_bin)
                    {
                        return -1;
                    }
                    int idx = static_cast<int>((mass - min_bin) / bin_width);
                    // 处理刚好等于max_bin的情况（概率很小）
                    if (idx >= n_bins)
                        idx = n_bins - 1;
                    return idx;
                });

            // 使用zip迭代器将bin索引和权重组合
            auto zipped_begin = thrust::make_zip_iterator(
                thrust::make_tuple(bin_indices.begin(), device_weights.begin()));
            auto zipped_end = thrust::make_zip_iterator(
                thrust::make_tuple(bin_indices.end(), device_weights.end()));

            // 移除无效的（bin索引为-1的）
            auto new_end = thrust::remove_if(
                zipped_begin,
                zipped_end,
                [=] __device__(const thrust::tuple<int, double> &t) -> bool
                {
                    return thrust::get<0>(t) < 0;
                });

            // 计算有效数量
            size_t valid_count = thrust::distance(zipped_begin, new_end);

            // 分离有效的bin索引和权重
            thrust::device_vector<int> valid_bins(valid_count);
            thrust::device_vector<double> valid_weights(valid_count);

            thrust::transform(
                zipped_begin, new_end,
                thrust::make_zip_iterator(thrust::make_tuple(
                    valid_bins.begin(), valid_weights.begin())),
                [=] __device__(const thrust::tuple<int, double> &t)
                {
                    return t;
                });

            // 按bin索引排序
            thrust::sort_by_key(valid_bins.begin(), valid_bins.end(), valid_weights.begin());

            // 使用reduce_by_key累加权重
            thrust::device_vector<int> unique_bins(n_bins);
            thrust::device_vector<double> bin_sums(n_bins);

            auto result = thrust::reduce_by_key(
                valid_bins.begin(), valid_bins.end(),
                valid_weights.begin(),
                unique_bins.begin(),
                bin_sums.begin());

            size_t num_unique = thrust::distance(unique_bins.begin(), result.first);

            // 填充结果
            for (size_t i = 0; i < num_unique; ++i)
            {
                int bin = unique_bins[i];
                if (bin >= 0 && bin < n_bins)
                {
                    bin_counts[bin] = bin_sums[i];
                }
            }
        }
        else
        {
            // 无权重的情况（更简单，使用直方图）
            thrust::device_vector<int> bin_indices(nEvents);

            // 计算每个质量值对应的bin索引
            thrust::transform(
                device_masses.begin(),
                device_masses.end(),
                bin_indices.begin(),
                [=] __device__(double mass) -> int
                {
                    if (mass < min_bin || mass >= max_bin)
                    {
                        return -1;
                    }
                    int idx = static_cast<int>((mass - min_bin) / bin_width);
                    return (idx < n_bins) ? idx : n_bins - 1;
                });

            // 移除超出范围的事件
            auto new_end = thrust::remove_if(
                bin_indices.begin(),
                bin_indices.end(),
                [] __device__(int idx) -> bool
                {
                    return idx < 0;
                });
            size_t valid_count = thrust::distance(bin_indices.begin(), new_end);

            // 对bin索引排序
            thrust::sort(bin_indices.begin(), bin_indices.begin() + valid_count);

            // 计算每个bin的事件数
            thrust::device_vector<int> unique_bins(n_bins);
            thrust::device_vector<int> bin_counts_int(n_bins);

            auto result = thrust::reduce_by_key(
                bin_indices.begin(),
                bin_indices.begin() + valid_count,
                thrust::constant_iterator<int>(1),
                unique_bins.begin(),
                bin_counts_int.begin());

            // 将int计数转换为double并复制到bin_counts
            size_t num_unique_bins = thrust::distance(unique_bins.begin(), result.first);
            for (size_t i = 0; i < num_unique_bins; ++i)
            {
                int bin = unique_bins[i];
                if (bin >= 0 && bin < n_bins)
                {
                    bin_counts[bin] = static_cast<double>(bin_counts_int[i]);
                }
            }
        }

        // 3.6 将结果复制回主机并填充TH1F
        thrust::host_vector<double> host_bin_counts = bin_counts;
        for (int bin = 0; bin < n_bins; ++bin)
        {
            // std::cout << "Bin " << bin << ": Count = " << host_bin_counts[bin] << std::endl;
            hist->SetBinContent(bin + 1, host_bin_counts[bin]);
        }

        // 3.7 设置正确的直方图范围
        hist->SetBins(n_bins, min_bin, max_bin);

        // std::cout << "Processed config " << configIdx << " with " << nEvents << " events" << std::endl;
    }
}

// 计算角度
__device__ __host__ double EvtDecayAngleDevice(const LorentzVector &p, const LorentzVector &q, const LorentzVector &d)
{
    double pd = p.Dot(d);
    double pq = p.Dot(q);
    double qd = q.Dot(d);
    double mp2 = p.M2();
    double mq2 = q.M2();
    double md2 = d.M2();

    double denominator = (pq * pq - mq2 * mp2) * (qd * qd - mq2 * md2);
    if (denominator <= 0.0)
        return 0.0; // Avoid sqrt of negative or zero

    double cost = (pd * mq2 - pq * qd) / sqrt(denominator);

    // Ensure cost is within [-1, 1] due to numerical errors
    if (cost > 1.0)
        cost = 1.0;
    if (cost < -1.0)
        cost = -1.0;

    return cost;
}

// Functor for calculating helicity angle
struct AngleCalculator
{
    const LorentzVector *momenta;
    const int *particle_indices;
    const int *group_sizes;
    int total_particles;

    __host__ __device__ double operator()(int event_idx) const
    {
        // particle_indices[0] = p, [1] = q, [2] = d
        // const LorentzVector &p = momenta[event_idx * total_particles + particle_indices[0]];
        // const LorentzVector &q = momenta[event_idx * total_particles + particle_indices[1]];
        // const LorentzVector &d = momenta[event_idx * total_particles + particle_indices[2]];

        int idx_offset = 0;

        LorentzVector p;
        for (int i = 0; i < group_sizes[0]; ++i)
        {
            const LorentzVector &pi = momenta[event_idx * total_particles + particle_indices[idx_offset + i]];
            p = p + pi;
        }

        LorentzVector q;
        idx_offset += group_sizes[0];
        for (int i = 0; i < group_sizes[1]; ++i)
        {
            const LorentzVector &qi = momenta[event_idx * total_particles + particle_indices[idx_offset + i]];
            q = q + qi;
        }

        LorentzVector d;
        idx_offset += group_sizes[1];
        for (int i = 0; i < group_sizes[2]; ++i)
        {
            const LorentzVector &di = momenta[event_idx * total_particles + particle_indices[idx_offset + i]];
            d = d + di;
        }

        return EvtDecayAngleDevice(p, q, d);
    }
};

void CalculateAngleHist(
    LorentzVector *device_momenta,
    const std::map<std::string, int> &particleToIndex,
    const std::vector<AngleHistConfig> &histConfigs,
    double *weights,
    std::vector<TH1F *> &outputHistograms,
    int nEvents, int nParticles)
{
    // 1. 检查输入
    if (outputHistograms.size() != histConfigs.size())
    {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size() << ")" << std::endl;
        return;
    }

    if (nEvents == 0)
    {
        std::cerr << "Warning: No events to process!" << std::endl;
        return;
    }

    // 2. 复制权重到设备（如果有的话）
    thrust::device_vector<double> device_weights;
    if (weights != nullptr)
    {
        // thrust::host_vector<double> host_weights(weights, weights + nEvents);
        // device_weights = host_weights;
        device_weights = thrust::device_vector<double>(weights, weights + nEvents);
    }

    // 3. 对每个直方图配置进行处理
    for (size_t configIdx = 0; configIdx < histConfigs.size(); ++configIdx)
    {
        const auto &config = histConfigs[configIdx];
        TH1F *hist = outputHistograms[configIdx];

        if (!hist)
        {
            std::cerr << "Error: outputHistograms[" << configIdx << "] is null!" << std::endl;
            continue;
        }

        hist->Reset();

        // 3.1 获取该配置需要的粒子索引和组大小
        std::vector<int> particleIndices;
        std::vector<int> groupSizes;

        for (const auto &particleGroup : config.particles)
        {
            groupSizes.push_back(particleGroup.size());
            for (const auto &particleName : particleGroup)
            {
                auto it = particleToIndex.find(particleName);
                if (it == particleToIndex.end())
                {
                    std::cerr << "Error: Particle '" << particleName
                              << "' not found in particleToIndex map!" << std::endl;
                    particleIndices.clear();
                    groupSizes.clear();
                    break;
                }
                particleIndices.push_back(it->second);
            }
            if (particleIndices.empty())
                break;
        }

        if (particleIndices.empty())
        {
            std::cerr << "Warning: Config " << configIdx << " has no valid particles!" << std::endl;
            continue;
        }

        if (groupSizes.size() < 2)
        {
            std::cerr << "Error: Config " << configIdx << " needs at least 2 particle groups for Dalitz plot!" << std::endl;
            continue;
        }

        // 检查range
        if (config.range.size() < 2)
        {
            std::cerr << "Error: Config " << configIdx << " range size < 2!" << std::endl;
            continue;
        }

        double min_bin = config.range[0];
        double max_bin = config.range[1];
        int n_bins = config.bins;
        double bin_width = (max_bin - min_bin) / n_bins;

        // 3.2 将粒子索引和组大小复制到设备
        thrust::device_vector<int> device_particle_indices = particleIndices;
        thrust::device_vector<int> device_group_sizes = groupSizes;
        int n_groups = groupSizes.size();

        // 3.3 为每个事件计算helicity角（使用Thrust transform）
        thrust::device_vector<double> device_angles(nEvents);

        // 获取设备原始指针
        // LorentzVector *d_momenta_ptr = h_momenta.momenta;
        int *d_indices_ptr = thrust::raw_pointer_cast(device_particle_indices.data());
        int *d_group_sizes_ptr = thrust::raw_pointer_cast(device_group_sizes.data());

        // 创建索引序列 [0, 1, 2, ..., nEvents-1]
        thrust::device_vector<int> event_indices(nEvents);
        thrust::sequence(event_indices.begin(), event_indices.end(), 0);

        // 计算所有事件的helicity角
        thrust::transform(
            event_indices.begin(),
            event_indices.end(),
            device_angles.begin(),
            AngleCalculator{
                device_momenta,
                d_indices_ptr,
                d_group_sizes_ptr,
                nParticles});

        // 3.4 创建直方图bin边界
        thrust::device_vector<double> bin_edges(n_bins + 1);
        for (int i = 0; i <= n_bins; ++i)
        {
            bin_edges[i] = min_bin + i * bin_width;
        }

        // 3.5 将角度分配到bins中（使用权重）
        thrust::device_vector<double> bin_counts(n_bins, 0.0);

        if (weights != nullptr)
        {
            // 有权重的情况
            thrust::device_vector<int> bin_indices(nEvents);

            // 使用lower_bound找到每个角度值对应的bin索引
            thrust::lower_bound(
                bin_edges.begin(),
                bin_edges.end() - 1, // 排除最后一个边界
                device_angles.begin(),
                device_angles.end(),
                bin_indices.begin());

            // 调整索引（从0开始，并且处理超出范围的情况）
            thrust::transform(
                bin_indices.begin(),
                bin_indices.end(),
                device_angles.begin(),
                bin_indices.begin(),
                [=] __device__(int idx, double angle) -> int
                {
                    if (angle < min_bin || angle >= max_bin)
                    {
                        return -1; // 超出范围
                    }
                    return idx; // idx已经是正确的bin索引
                });

            // 使用reduce_by_key来累加权重
            // 首先需要排序bin_indices以便使用reduce_by_key
            thrust::device_vector<int> sorted_bin_indices = bin_indices;
            thrust::device_vector<double> event_weights = device_weights;

            // 按bin索引排序
            thrust::sort_by_key(sorted_bin_indices.begin(), sorted_bin_indices.end(), event_weights.begin());

            // 移除超出范围的事件
            // 首先计算有效事件数
            size_t valid_count = thrust::count_if(
                sorted_bin_indices.begin(),
                sorted_bin_indices.end(),
                [] __device__(int idx) -> bool
                {
                    return idx >= 0;
                });

            // 将有效元素移动到序列开头
            auto new_end_indices = thrust::remove_if(
                sorted_bin_indices.begin(),
                sorted_bin_indices.end(),
                [] __device__(int idx) -> bool
                {
                    return idx < 0;
                });

            auto new_end_weights = thrust::remove_if(
                event_weights.begin(),
                event_weights.end(),
                sorted_bin_indices.begin(),
                [] __device__(int idx) -> bool
                {
                    return idx < 0;
                });

            // 调整大小以仅包含有效元素
            sorted_bin_indices.resize(valid_count);
            event_weights.resize(valid_count);

            // 分配临时数组用于reduce_by_key
            thrust::device_vector<int> unique_bins(n_bins);
            thrust::device_vector<double> bin_sums(n_bins);

            // 对相同bin的事件权重求和
            thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<double>::iterator> new_end_pair;
            new_end_pair = thrust::reduce_by_key(
                sorted_bin_indices.begin(),
                sorted_bin_indices.begin() + valid_count,
                event_weights.begin(),
                unique_bins.begin(),
                bin_sums.begin());

            // 将结果复制回bin_counts
            size_t num_unique_bins = thrust::distance(unique_bins.begin(), new_end_pair.first);
            for (size_t i = 0; i < num_unique_bins; ++i)
            {
                int bin = unique_bins[i];
                if (bin >= 0 && bin < n_bins)
                {
                    bin_counts[bin] = bin_sums[i];
                }
            }
        }
        else
        {
            // 无权重的情况（更简单，使用直方图）
            thrust::device_vector<int> bin_indices(nEvents);

            // 计算每个角度值对应的bin索引
            thrust::transform(
                device_angles.begin(),
                device_angles.end(),
                bin_indices.begin(),
                [=] __device__(double angle) -> int
                {
                    if (angle < min_bin || angle >= max_bin)
                    {
                        return -1;
                    }
                    int idx = static_cast<int>((angle - min_bin) / bin_width);
                    return (idx < n_bins) ? idx : n_bins - 1;
                });

            // 移除超出范围的事件
            auto new_end = thrust::remove_if(
                bin_indices.begin(),
                bin_indices.end(),
                [] __device__(int idx) -> bool
                {
                    return idx < 0;
                });
            size_t valid_count = thrust::distance(bin_indices.begin(), new_end);

            // 对bin索引排序
            thrust::sort(bin_indices.begin(), bin_indices.begin() + valid_count);

            // 计算每个bin的事件数
            thrust::device_vector<int> unique_bins(n_bins);
            thrust::device_vector<int> bin_counts_int(n_bins);

            auto result = thrust::reduce_by_key(
                bin_indices.begin(),
                bin_indices.begin() + valid_count,
                thrust::constant_iterator<int>(1),
                unique_bins.begin(),
                bin_counts_int.begin());

            // 将int计数转换为double并复制到bin_counts
            size_t num_unique_bins = thrust::distance(unique_bins.begin(), result.first);
            for (size_t i = 0; i < num_unique_bins; ++i)
            {
                int bin = unique_bins[i];
                if (bin >= 0 && bin < n_bins)
                {
                    bin_counts[bin] = static_cast<double>(bin_counts_int[i]);
                }
            }
        }

        // 3.6 将结果复制回主机并填充TH1F
        thrust::host_vector<double> host_bin_counts = bin_counts;
        for (int bin = 0; bin < n_bins; ++bin)
        {
            // std::cout << "Bin " << bin << ": Count = " << host_bin_counts[bin] << std::endl;
            hist->SetBinContent(bin + 1, host_bin_counts[bin]);
        }

        // 3.7 设置正确的直方图范围
        hist->SetBins(n_bins, min_bin, max_bin);

        // std::cout << "Processed helicity config " << configIdx << " with " << nEvents << " events" << std::endl;
    }
}

// 计算Dalitz图
struct DalitzCalculator
{
    const LorentzVector *momenta;
    const int *particle_indices; // Array of 3 particle group indices: [group1_idx1, group1_idx2, ..., group2_idx1, ...]
    const int *group_sizes;      // Size of each particle group
    int n_groups;                // Number of particle groups (should be 2 or 3)
    int total_particles;

    __host__ __device__ thrust::pair<double, double> operator()(int event_idx) const
    {
        double m12_sq = 0.0;
        double m13_sq = 0.0;

        // Calculate invariant mass squared for first two groups
        int idx_offset = 0;

        // Group 1
        LorentzVector total1;
        for (int i = 0; i < group_sizes[0]; ++i)
        {
            const LorentzVector &p = momenta[event_idx * total_particles + particle_indices[idx_offset + i]];
            total1 = total1 + p;
        }
        m12_sq = total1.M2();
        idx_offset += group_sizes[0];

        // Group 2
        LorentzVector total2;
        for (int i = 0; i < group_sizes[1]; ++i)
        {
            const LorentzVector &p = momenta[event_idx * total_particles + particle_indices[idx_offset + i]];
            total2 = total2 + p;
        }
        m13_sq = total2.M2();

        return thrust::make_pair(m12_sq, m13_sq);
    }
};

void CalculateDalitzHist(
    LorentzVector *device_momenta,
    const std::map<std::string, int> &particleToIndex,
    const std::vector<DalitzHistConfig> &histConfigs,
    double *weights,
    std::vector<TH2F *> &outputHistograms,
    int nEvents, int nParticles)
{
    // 1. 检查输入
    if (outputHistograms.size() != histConfigs.size())
    {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size() << ")" << std::endl;
        return;
    }

    if (nEvents == 0)
    {
        std::cerr << "Warning: No events to process!" << std::endl;
        return;
    }

    // 2. 复制权重到设备（如果有的话）
    thrust::device_vector<double> device_weights;
    if (weights != nullptr)
    {
        // thrust::host_vector<double> host_weights(weights, weights + nEvents);
        // device_weights = host_weights;
        device_weights = thrust::device_vector<double>(weights, weights + nEvents);
    }

    // 3. 对每个直方图配置进行处理
    for (size_t configIdx = 0; configIdx < histConfigs.size(); ++configIdx)
    {
        const auto &config = histConfigs[configIdx];
        TH2F *hist = outputHistograms[configIdx];

        if (!hist)
        {
            std::cerr << "Error: outputHistograms[" << configIdx << "] is null!" << std::endl;
            continue;
        }

        hist->Reset();

        // 3.1 获取该配置需要的粒子索引和组大小
        std::vector<int> particleIndices;
        std::vector<int> groupSizes;

        for (const auto &particleGroup : config.particles)
        {
            groupSizes.push_back(particleGroup.size());
            for (const auto &particleName : particleGroup)
            {
                auto it = particleToIndex.find(particleName);
                if (it == particleToIndex.end())
                {
                    std::cerr << "Error: Particle '" << particleName
                              << "' not found in particleToIndex map!" << std::endl;
                    particleIndices.clear();
                    groupSizes.clear();
                    break;
                }
                particleIndices.push_back(it->second);
            }
            if (particleIndices.empty())
                break;
        }

        if (particleIndices.empty())
        {
            std::cerr << "Warning: Config " << configIdx << " has no valid particles!" << std::endl;
            continue;
        }

        if (groupSizes.size() < 2)
        {
            std::cerr << "Error: Config " << configIdx << " needs at least 2 particle groups for Dalitz plot!" << std::endl;
            continue;
        }

        // 检查range和bins
        if (config.range.size() < 2 || config.bins.size() < 2)
        {
            std::cerr << "Error: Config " << configIdx << " range or bins size < 2!" << std::endl;
            continue;
        }

        double x_min = config.range[0][0];
        double x_max = config.range[0][1];
        double y_min = config.range[1][0];
        double y_max = config.range[1][1];
        int x_bins = config.bins[0];
        int y_bins = config.bins[1];

        double x_bin_width = (x_max - x_min) / x_bins;
        double y_bin_width = (y_max - y_min) / y_bins;

        // 3.2 将粒子索引和组大小复制到设备
        thrust::device_vector<int> device_particle_indices = particleIndices;
        thrust::device_vector<int> device_group_sizes = groupSizes;
        int n_groups = groupSizes.size();

        // 3.3 为每个事件计算Dalitz坐标（使用Thrust transform）
        thrust::device_vector<thrust::pair<double, double>> device_coords(nEvents);

        // 获取设备原始指针
        // LorentzVector *d_momenta_ptr = h_momenta.momenta;
        int *d_indices_ptr = thrust::raw_pointer_cast(device_particle_indices.data());
        int *d_group_sizes_ptr = thrust::raw_pointer_cast(device_group_sizes.data());

        // 创建索引序列 [0, 1, 2, ..., nEvents-1]
        thrust::device_vector<int> event_indices(nEvents);
        thrust::sequence(event_indices.begin(), event_indices.end(), 0);

        // 计算所有事件的Dalitz坐标
        thrust::transform(
            event_indices.begin(),
            event_indices.end(),
            device_coords.begin(),
            DalitzCalculator{
                device_momenta,
                d_indices_ptr,
                d_group_sizes_ptr,
                n_groups,
                nParticles});

        // 3.4 将坐标复制回主机
        thrust::host_vector<thrust::pair<double, double>> host_coords = device_coords;

        // 3.5 填充二维直方图
        hist->SetBins(x_bins, x_min, x_max, y_bins, y_min, y_max);

        if (weights != nullptr)
        {
            // 有权重的情况
            thrust::host_vector<double> host_weights = device_weights;
            for (int evt = 0; evt < nEvents; ++evt)
            {
                double x = host_coords[evt].first;
                double y = host_coords[evt].second;

                // 检查坐标是否在范围内
                if (x >= x_min && x < x_max && y >= y_min && y < y_max)
                {
                    hist->Fill(x, y, host_weights[evt]);
                }
            }
        }
        else
        {
            // 无权重的情况
            for (int evt = 0; evt < nEvents; ++evt)
            {
                double x = host_coords[evt].first;
                double y = host_coords[evt].second;

                if (x >= x_min && x < x_max && y >= y_min && y < y_max)
                {
                    hist->Fill(x, y);
                }
            }
        }

        // std::cout << "Processed Dalitz config " << configIdx << " with " << nEvents << " events" << std::endl;
    }
}
