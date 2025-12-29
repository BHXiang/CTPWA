#include <Figure.cuh>
#include <helicity.cuh>

// 批量计算所有直方图（单线程版本）
void CalculateMassHist(std::map<std::string, std::vector<LorentzVector>> &momenta, std::vector<MassHistConfig> &histConfigs, double *weight, std::vector<TH1F *> &outputHistograms)
{
    if (outputHistograms.size() != histConfigs.size())
    {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size() << ")" << std::endl;
        return;
    }

    size_t nEvents = 0;
    if (!momenta.empty())
    {
        nEvents = momenta.begin()->second.size();
    }

    for (size_t i = 0; i < histConfigs.size(); ++i)
    {
        const auto &config = histConfigs[i];
        TH1F *hist = outputHistograms[i];

        if (!hist)
        {
            std::cerr << "Error: outputHistograms[" << i << "] is null!" << std::endl;
            continue;
        }

        hist->Reset();

        for (size_t evt = 0; evt < nEvents; ++evt)
        {
            bool allFound = true;

            double mass = 0.0;
            LorentzVector total = LorentzVector();
            for (const auto &particleName : config.particles)
            {
                if (momenta.count(particleName) && evt < momenta.at(particleName).size())
                {
                    const auto &p = momenta.at(particleName)[evt];
                    total = total + p;
                }
                else
                {
                    allFound = false;
                    break;
                }
            }

            if (allFound)
            {
                if (weight != nullptr)
                    hist->Fill(total.M(), weight[evt]);
                else
                    hist->Fill(total.M());
            }
        }
    }
}

void CalculateAngleHist(std::map<std::string, std::vector<LorentzVector>> &momenta, std::vector<AngleHistConfig> &histConfigs, double *weight, std::vector<TH1F *> &outputHistograms)
{
    if (outputHistograms.size() != histConfigs.size())
    {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size() << ")" << std::endl;
        return;
    }

    size_t nEvents = 0;
    if (!momenta.empty())
    {
        nEvents = momenta.begin()->second.size();
    }

    for (size_t i = 0; i < histConfigs.size(); ++i)
    {
        const auto &config = histConfigs[i];
        TH1F *hist = outputHistograms[i];

        if (!hist)
        {
            std::cerr << "Error: outputHistograms[" << i << "] is null!" << std::endl;
            continue;
        }

        hist->Reset();

        for (size_t evt = 0; evt < nEvents; ++evt)
        {
            bool allFound = true;

            double angle = 0.0;
            // 修复1: 重命名vector以避免冲突
            std::vector<TLorentzVector> lorentzVectors;

            // 修复2: 使用不同的循环变量名
            for (const auto &particleGroup : config.particles)
            {
                LorentzVector combined = LorentzVector();

                // std::cout << "Combining particles: ";
                for (const auto &particleName : particleGroup)
                {
                    // std::cout << particleName << " ";

                    const auto &p = momenta.at(particleName)[evt];
                    combined = combined + p;
                }
                // std::cout << std::endl;

                // 修复3: 使用正确的vector
                lorentzVectors.push_back(TLorentzVector(combined.Px, combined.Py, combined.Pz, combined.E));
            }

            // 确保有足够的向量进行变换
            if (lorentzVectors.size() >= 3)
            {
                lorentzVectors[1].Boost(-lorentzVectors[0].BoostVector());
                lorentzVectors[2].Boost(-lorentzVectors[0].BoostVector());
                lorentzVectors[2].Boost(-lorentzVectors[1].BoostVector());
                lorentzVectors[2].RotateZ(-lorentzVectors[1].Phi());
                lorentzVectors[2].RotateY(-lorentzVectors[1].Theta());
                angle = lorentzVectors[2].CosTheta();

                if (allFound)
                {
                    if (weight != nullptr)
                        hist->Fill(angle, weight[evt]);
                    else
                        hist->Fill(angle);
                }
            }
            else
            {
                std::cerr << "Warning: Not enough particle groups for transformation." << std::endl;
            }
        }
    }
}

void CalculateDalitzHist(std::map<std::string, std::vector<LorentzVector>> &momenta, std::vector<DalitzHistConfig> &histConfigs, double *weight, std::vector<TH2F *> &outputHistograms)
{
    if (outputHistograms.size() != histConfigs.size())
    {
        std::cerr << "Error: outputHistograms size (" << outputHistograms.size()
                  << ") does not match histConfigs size (" << histConfigs.size() << ")" << std::endl;
        return;
    }

    size_t nEvents = 0;
    if (!momenta.empty())
    {
        nEvents = momenta.begin()->second.size();
    }

    for (size_t i = 0; i < histConfigs.size(); ++i)
    {
        const auto &config = histConfigs[i];
        TH2F *hist = outputHistograms[i];

        if (!hist)
        {
            std::cerr << "Error: outputHistograms[" << i << "] is null!" << std::endl;
            continue;
        }

        hist->Reset();

        // 检查配置是否适合Dalitz图
        if (config.particles.size() < 2)
        {
            std::cerr << "Error: DalitzHistConfig[" << i << "] needs at least 2 particle groups for Dalitz plot!" << std::endl;
            continue;
        }

        for (size_t evt = 0; evt < nEvents; ++evt)
        {
            bool allFound = true;
            std::vector<double> masses; // 存储每个粒子组合的不变质量

            // 计算每个粒子组合的不变质量
            for (const auto &particleGroup : config.particles)
            {
                LorentzVector total = LorentzVector();

                for (const auto &particleName : particleGroup)
                {
                    if (momenta.count(particleName) && evt < momenta.at(particleName).size())
                    {
                        const auto &p = momenta.at(particleName)[evt];
                        total = total + p;
                    }
                    else
                    {
                        allFound = false;
                        break;
                    }
                }

                if (!allFound)
                    break;
                masses.push_back(total.M());
            }

            if (allFound && masses.size() >= 2)
            {
                // 使用前两个粒子组合的质量作为Dalitz图的X和Y轴
                double m12_sq = masses[0] * masses[0]; // m12^2
                double m13_sq = masses[1] * masses[1]; // m13^2

                // 也可以计算第三个组合的质量，用于检查
                if (config.particles.size() >= 3)
                {
                    double m23_sq = masses[2] * masses[2]; // m23^2
                    // 如果需要，可以用第三个质量进行约束检查
                }

                if (weight != nullptr)
                    hist->Fill(m12_sq, m13_sq, weight[evt]);
                else
                    hist->Fill(m12_sq, m13_sq);
            }
        }
    }
}
