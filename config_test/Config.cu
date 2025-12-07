#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <map>
#include <array>
#include <iostream>
#include <iomanip>

struct ParticleConfig
{
    std::string name;
    int J;
    int P;
    double mass;
};

struct ResonanceConfig
{
    std::string name;
    std::string type;
    std::vector<double> parameters;
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

    const std::vector<ParticleConfig> &getParticles() const { return particles_; }
    const std::vector<DecayChainConfig> &getDecayChains() const { return decay_chains_; }
    const std::map<std::string, ResonanceConfig> &getResonances() const { return resonances_; }
    const std::vector<std::pair<std::string, std::string>> &getConjugatePairs() const { return conjugate_pairs_; }
    const std::map<std::string, std::vector<std::string>> &getDataFiles() const { return data_files_; }
    const std::vector<std::string> &getDatOrder() const { return dat_order_; }

private:
    void parseParticles(const YAML::Node &node)
    {
        for (const auto &particle_node : node)
        {
            std::string name = particle_node.first.as<std::string>();
            const auto &props = particle_node.second;

            ParticleConfig particle;
            particle.name = name;
            particle.J = props["J"].as<int>();
            particle.P = props["P"].as<int>();
            particle.mass = props["mass"].as<double>();

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
                if (key != "decay")
                {
                    ResonanceChainConfig res_chain;
                    res_chain.intermediate = key;

                    for (const auto &spin_config : res_chain_node.second)
                    {
                        SpinChainConfig spin_chain;
                        spin_chain.spin_parity = spin_config["spin_parity"].as<std::vector<int>>();
                        spin_chain.resonances = spin_config["resonances"].as<std::vector<std::string>>();
                        res_chain.spin_chains.push_back(spin_chain);
                    }

                    chain.resonance_chains.push_back(res_chain);
                }
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
            res.type = props["type"].as<std::string>();
            res.parameters = props["parameters"].as<std::vector<double>>();

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

    std::vector<ParticleConfig> particles_;
    std::vector<DecayChainConfig> decay_chains_;
    std::map<std::string, ResonanceConfig> resonances_;
    std::vector<std::pair<std::string, std::string>> conjugate_pairs_;
    std::map<std::string, std::vector<std::string>> data_files_;
    std::vector<std::string> dat_order_;
};

// Main function to test config reading
int main()
{
    try
    {
        std::cout << "=== Testing Config Parser ===" << std::endl;
        std::cout << "Reading config.yml..." << std::endl;

        // Create config parser
        ConfigParser parser("config.yml");

        std::cout << "\n=== Particles ===" << std::endl;
        const auto &particles = parser.getParticles();
        for (const auto &particle : particles)
        {
            std::cout << "Name: " << particle.name
                      << ", J: " << particle.J
                      << ", P: " << particle.P
                      << ", mass: " << particle.mass << std::endl;
        }

        std::cout << "\n=== Data Files ===" << std::endl;
        const auto &data_files = parser.getDataFiles();
        for (const auto &[type, files] : data_files)
        {
            std::cout << type << ": " << std::endl;
            for (const auto &file : files)
            {
                std::cout << "  - " << file << std::endl;
            }
        }

        std::cout << "\n=== Data Order ===" << std::endl;
        const auto &dat_order = parser.getDatOrder();
        for (const auto &particle : dat_order)
        {
            std::cout << "  - " << particle << std::endl;
        }

        std::cout << "\n=== Decay Chains ===" << std::endl;
        const auto &decay_chains = parser.getDecayChains();
        for (const auto &chain : decay_chains)
        {
            std::cout << "Chain: " << chain.name << std::endl;

            for (size_t i = 0; i < chain.decay_steps.size(); ++i)
            {
                const auto &step = chain.decay_steps[i];
                std::cout << "  Decay Step " << (i + 1) << ": " << step.mother << " -> ";
                for (const auto &daughter : step.daughters)
                {
                    std::cout << daughter << " ";
                }
                std::cout << std::endl;
            }

            for (const auto &res_chain : chain.resonance_chains)
            {
                std::cout << "  Intermediate: " << res_chain.intermediate << std::endl;
                for (const auto &spin_chain : res_chain.spin_chains)
                {
                    std::cout << "    Spin/Parity: [";
                    for (size_t i = 0; i < spin_chain.spin_parity.size(); ++i)
                    {
                        std::cout << spin_chain.spin_parity[i];
                        if (i < spin_chain.spin_parity.size() - 1)
                            std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                    std::cout << "    Resonances: ";
                    for (const auto &res : spin_chain.resonances)
                    {
                        std::cout << res << " ";
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }

        std::cout << "\n=== Resonances ===" << std::endl;
        const auto &resonances = parser.getResonances();
        for (const auto &[name, res] : resonances)
        {
            std::cout << "Name: " << name
                      << ", Type: " << res.type
                      << ", Parameters: [";
            for (size_t i = 0; i < res.parameters.size(); ++i)
            {
                std::cout << res.parameters[i];
                if (i < res.parameters.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "\n=== Conjugate Pairs ===" << std::endl;
        const auto &conjugate_pairs = parser.getConjugatePairs();
        for (const auto &pair : conjugate_pairs)
        {
            std::cout << pair.first << " <-> " << pair.second << std::endl;
        }

        std::cout << "\n=== Test Completed Successfully ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}