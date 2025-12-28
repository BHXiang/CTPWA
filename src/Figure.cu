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
