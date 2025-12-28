#ifndef FIGURE_CUH
#define FIGURE_CUH

#include <helicity.cuh>

#include <vector>
#include <map>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TROOT.h>

// 直方图配置结构体（包含直方图对象）
struct MassHistConfig
{
    std::string name;
    std::string title;
    std::vector<std::string> particles;
    int bins;
    std::vector<double> range;
    std::vector<std::string> tex;

    MassHistConfig(const std::string &n, const std::string &t,
                   const std::vector<std::string> &p,
                   int b, const std::vector<double> &r, const std::vector<std::string> &te = {})
        : name(n), title(t), particles(p), bins(b), range(r), tex(te) {}
};

struct AngleHistConfig
{
    std::string name;
    std::string title;
    std::vector<std::vector<std::string>> particles;
    int bins;
    std::vector<double> range;
    std::vector<std::string> tex;

    AngleHistConfig(const std::string &n, const std::string &t,
                    const std::vector<std::vector<std::string>> &p,
                    int b, const std::vector<double> &r, const std::vector<std::string> &te = {})
        : name(n), title(t), particles(p), bins(b), range(r), tex(te) {}
};

struct DalitzHistConfig
{
    std::string name;
    std::string title;
    std::vector<std::vector<std::string>> particles;
    std::vector<int> bins;
    std::vector<double> range;
    std::vector<std::string> tex;

    DalitzHistConfig(const std::string &n, const std::string &t,
                     const std::vector<std::vector<std::string>> &p,
                     const std::vector<int> &b, const std::vector<double> &r, const std::vector<std::string> &te = {})
        : name(n), title(t), particles(p), bins(b), range(r), tex(te) {}
};

void CalculateMassHist(std::map<std::string, std::vector<LorentzVector>> &momenta, std::vector<MassHistConfig> &histConfigs, double *weight, std::vector<TH1F *> &outputHistograms);
void CalculateAngleHist(std::map<std::string, std::vector<LorentzVector>> &momenta, std::vector<AngleHistConfig> &histConfigs, double *weight, std::vector<TH1F *> &outputHistograms);
void CalculateDalitzHist(std::map<std::string, std::vector<LorentzVector>> &momenta, std::vector<DalitzHistConfig> &histConfigs, double *weight, std::vector<TH2F *> &outputHistograms);

#endif // FIGURE_CUH