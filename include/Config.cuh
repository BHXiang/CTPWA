#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <AmpGen.cuh>
#include <string>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>

// struct Particle
// {
//     std::string name;
//     int spin;
//     int parity;
//     double mass;
//     std::string tex;
// };
// 粒子信息结构体
struct Particle
{
    std::string name;
    int spin;
    int parity;
    double mass;
    std::string tex;
};

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
    ConfigParser(const std::string &config_file);

    const std::vector<Particle> &getParticles() const { return particles_; }
    const std::vector<DecayChainConfig> &getDecayChains() const { return decay_chains_; }
    const std::map<std::string, ResonanceConfig> &getResonances() const { return resonances_; }
    const std::vector<std::pair<std::string, std::string>> &getConjugatePairs() const { return conjugate_pairs_; }
    const std::map<std::string, std::vector<std::string>> &getDataFiles() const { return data_files_; }
    const std::vector<std::string> &getDatOrder() const { return dat_order_; }

    // 定制legend功能
    std::vector<std::string> getCustomLegends() const;
    std::map<std::string, std::vector<std::string>> getChainCustomLegends() const;

    // 生成Legend的函数
    std::string generateLegend(const std::vector<std::string> &particles) const;

    // 获取单个粒子的Tex
    std::string getParticleTex(const std::string &name) const;

    // 获取单个共振态的Tex
    std::string getResonanceTex(const std::string &name) const;

    // 按decay chain顺序获取所有legend
    std::vector<std::string> getAllLegends() const;

    // 获取所有可能的共振态组合的legend
    std::vector<std::string> getAllResonanceCombinationLegends() const;

private:
    // 生成定制legend：根据legend模板生成所有可能的组合
    std::vector<std::string> generateCustomLegends() const;

    // 获取每个decay chain的定制legend
    std::map<std::string, std::vector<std::string>> generateChainCustomLegends() const;

    // 解析函数
    void parseParticles(const YAML::Node &node);
    void parseData(const YAML::Node &node);
    void parseDecayChains(const YAML::Node &node);
    void parseResonances(const YAML::Node &node);
    void parseConjugatePairs(const YAML::Node &node);

    std::vector<Particle> particles_;
    std::vector<DecayChainConfig> decay_chains_;
    std::map<std::string, ResonanceConfig> resonances_;
    std::vector<std::pair<std::string, std::string>> conjugate_pairs_;
    std::map<std::string, std::vector<std::string>> data_files_;
    std::vector<std::string> dat_order_;
};

#endif // CONFIG_CUH