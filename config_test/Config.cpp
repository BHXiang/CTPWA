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
                                            [&](const ParticleConfig &p)
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
                                        [&](const ParticleConfig &p)
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
            std::vector<std::vector<ParticleConfig>> particleLists;

            for (const auto &res_chain : chain.resonance_chains)
            {
                std::vector<ParticleConfig> particles;
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
            std::vector<std::vector<ParticleConfig>> result = {{}};
            for (const auto &particleList : particleLists)
            {
                std::vector<std::vector<ParticleConfig>> temp;
                for (const auto &res : result)
                {
                    for (const auto &particle : particleList)
                    {
                        std::vector<ParticleConfig> new_res = res;
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
                    std::pair<std::string, std::vector<int>> key = {particle.name, {particle.J, particle.P}};
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

            ParticleConfig particle;
            particle.name = name;
            particle.J = props["J"].as<int>();
            particle.P = props["P"].as<int>();
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
        // ConfigParser parser("config_4body.yml");

        std::cout << "\n=== Particles ===" << std::endl;
        const auto &particles = parser.getParticles();
        for (const auto &particle : particles)
        {
            std::cout << "Name: " << particle.name
                      << ", J: " << particle.J
                      << ", P: " << particle.P
                      << ", mass: " << particle.mass
                      << ", Tex: " << particle.tex << std::endl;
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

            // 首先，我们需要构建一个映射：从中间态名称和自旋宇称到共振态列表的映射
            std::map<std::pair<std::string, std::vector<int>>, std::vector<std::string>> spin_resonance_map;
            // std::map<ParticleConfig, std::vector<std::string>> spin_resonance_map;
            std::vector<std::vector<ParticleConfig>> particleLists;

            for (const auto &res_chain : chain.resonance_chains)
            {
                std::vector<ParticleConfig> particles;
                for (const auto &spin_chain : res_chain.spin_chains)
                {
                    // ParticleConfig key; // = {res_chain.intermediate, spin_chain.spin_parity[0], spin_chain.spin_parity[1], -1.0};
                    // key.name = res_chain.intermediate;
                    // key.J = spin_chain.spin_parity[0];
                    // key.P = spin_chain.spin_parity[1];
                    std::pair<std::string, std::vector<int>> key = {res_chain.intermediate, {spin_chain.spin_parity[0], spin_chain.spin_parity[1]}}; //
                    spin_resonance_map[key] = spin_chain.resonances;
                    particles.push_back({res_chain.intermediate, spin_chain.spin_parity[0], spin_chain.spin_parity[1], -1.0});
                }
                particleLists.push_back(particles);
            }

            // 生成并输出所有组合
            std::vector<std::vector<ParticleConfig>> result = {{}};
            for (const auto &particleList : particleLists)
            {
                std::vector<std::vector<ParticleConfig>> temp;
                for (const auto &res : result)
                {
                    for (const auto &particle : particleList)
                    {
                        std::vector<ParticleConfig> new_res = res;
                        new_res.push_back(particle);
                        temp.push_back(new_res);
                    }
                }
                result = std::move(temp);
            }

            // 现在为每个JP组合生成共振态组合
            std::cout << "\n=== All Combinations (JP + Resonance) ===" << std::endl;
            int total_combinations = 0;

            for (size_t i = 0; i < result.size(); ++i)
            {
                const auto &jp_combination = result[i];

                std::cout << "\nJP Combination " << i + 1 << ": ";
                for (size_t j = 0; j < jp_combination.size(); ++j)
                {
                    const auto &p = jp_combination[j];
                    std::cout << p.name << "(" << p.J;
                    if (p.P == 1)
                        std::cout << "+";
                    else if (p.P == -1)
                        std::cout << "-";
                    else
                        std::cout << "^" << p.P;
                    std::cout << ")";
                    if (j < jp_combination.size() - 1)
                        std::cout << " + ";
                }
                std::cout << std::endl;

                // 为当前JP组合生成所有共振态组合
                std::vector<std::vector<std::pair<std::string, std::string>>> resonance_combinations = {{}};

                for (const auto &particle : jp_combination)
                {
                    // 获取当前中间态和自旋宇称对应的共振态列表
                    std::pair<std::string, std::vector<int>> key = {particle.name, {particle.J, particle.P}};
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

                // 输出当前JP组合下的所有共振态组合
                std::cout << "  Resonance combinations for this JP:" << std::endl;
                for (size_t k = 0; k < resonance_combinations.size(); ++k)
                {
                    std::cout << "    " << k + 1 << ". ";
                    for (size_t j = 0; j < resonance_combinations[k].size(); ++j)
                    {
                        const auto &res_pair = resonance_combinations[k][j];
                        std::cout << res_pair.second; // 共振态名称
                        if (j < resonance_combinations[k].size() - 1)
                            std::cout << " + ";
                    }
                    std::cout << std::endl;
                    total_combinations++;
                }

                std::cout << "  Total resonance combinations for this JP: " << resonance_combinations.size() << std::endl;
            }

            std::cout << "\n=== Summary ===" << std::endl;
            std::cout << "Total JP combinations: " << result.size() << std::endl;
            std::cout << "Total resonance combinations: " << total_combinations << std::endl;

            // 测试Legend功能
            std::cout << "\n=== Legend Examples ===" << std::endl;
            // 示例1: [R_KK, eta]
            std::vector<std::string> example1 = {"R_KK", "eta"};
            std::cout << "Legend for [R_KK, eta]: " << parser.generateLegend(example1) << std::endl;

            // 示例2: [phi1020, eta]
            std::vector<std::string> example2 = {"phi1020", "eta"};
            std::cout << "Legend for [phi1020, eta]: " << parser.generateLegend(example2) << std::endl;

            // std::vector<std::vector<ParticleConfig>> particleLists;
            // for (const auto &res_chain : chain.resonance_chains)
            // {
            //     std::vector<ParticleConfig> particles;
            //     for (const auto &spin_chain : res_chain.spin_chains)
            //     {
            //         particles.push_back({
            //             res_chain.intermediate,
            //             static_cast<int>(spin_chain.spin_parity[0]),
            //             static_cast<int>(spin_chain.spin_parity[1]),
            //             -1 // mass
            //         });
            //     }
            //     particleLists.push_back(particles);
            // }

            // // 生成并输出所有组合
            // std::vector<std::vector<ParticleConfig>> result = {{}};
            // for (const auto &particleList : particleLists)
            // {
            //     std::vector<std::vector<ParticleConfig>> temp;
            //     for (const auto &res : result)
            //     {
            //         for (const auto &particle : particleList)
            //         {
            //             std::vector<ParticleConfig> new_res = res;
            //             new_res.push_back(particle);
            //             temp.push_back(new_res);
            //         }
            //     }
            //     result = std::move(temp);
            // }

            // // 输出结果
            // for (size_t i = 0; i < result.size(); ++i)
            // {
            //     std::cout << "Combination " << i + 1 << ": ";
            //     for (size_t j = 0; j < result[i].size(); ++j)
            //     {
            //         const auto &p = result[i][j];
            //         std::cout << p.name << "(" << p.J;
            //         if (p.P == 1)
            //             std::cout << "+";
            //         else if (p.P == -1)
            //             std::cout << "-";
            //         else
            //             std::cout << "^" << p.P;
            //         std::cout << ")";
            //         if (j < result[i].size() - 1)
            //             std::cout << " -> ";
            //     }
            //     std::cout << std::endl;
            // }
        }

        std::cout << "\n=== Resonances ===" << std::endl;
        const auto &resonances = parser.getResonances();
        for (const auto &[name, res] : resonances)
        {
            std::cout << "Name: " << name
                      << ", J: " << res.J
                      << ", P: " << res.P
                      << ", Model: " << res.type
                      << ", Parameters: [";
            for (size_t i = 0; i < res.parameters.size(); ++i)
            {
                std::cout << res.parameters[i];
                if (i < res.parameters.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]"
                      << ", Tex: " << res.tex << std::endl;
        }

        std::cout << "\n=== Conjugate Pairs ===" << std::endl;
        const auto &conjugate_pairs = parser.getConjugatePairs();
        for (const auto &pair : conjugate_pairs)
        {
            std::cout << pair.first << " <-> " << pair.second << std::endl;
        }

        // // 测试新的legend功能
        // std::cout << "\n=== Enhanced Legend Functions ===" << std::endl;

        // // 测试单个粒子的Tex
        // std::cout << "\n--- Single Particle Tex ---" << std::endl;
        // std::cout << "Jpsi Tex: " << parser.getParticleTex("Jpsi") << std::endl;
        // std::cout << "eta Tex: " << parser.getParticleTex("eta") << std::endl;
        // std::cout << "Kp Tex: " << parser.getParticleTex("Kp") << std::endl;

        // // 测试单个共振态的Tex
        // std::cout << "\n--- Single Resonance Tex ---" << std::endl;
        // std::cout << "phi1020 Tex: " << parser.getResonanceTex("phi1020") << std::endl;
        // std::cout << "K1_1410p Tex: " << parser.getResonanceTex("K1_1410p") << std::endl;

        // // 测试未知粒子的Tex（字符输出）
        // std::cout << "\n--- Unknown Particle Tex (Character Output) ---" << std::endl;
        // std::cout << "Unknown Tex: " << parser.getParticleTex("UnknownParticle") << std::endl;

        // // 测试按decay chain顺序的所有legend
        // std::cout << "\n--- All Legends by Decay Chain Order ---" << std::endl;
        // const auto &all_legends = parser.getAllLegends();
        // for (size_t i = 0; i < all_legends.size(); ++i)
        // {
        //     std::cout << i + 1 << ". " << all_legends[i] << std::endl;
        // }

        // // 测试所有可能的共振态组合的legend
        // std::cout << "\n--- All Resonance Combination Legends ---" << std::endl;
        // const auto &resonance_legends = parser.getAllResonanceCombinationLegends();
        // std::cout << "Total resonance combination legends: " << resonance_legends.size() << std::endl;

        // // 显示前20个共振态组合legend
        // for (size_t i = 0; i < std::min(resonance_legends.size(), size_t(20)); ++i)
        // {
        //     std::cout << i + 1 << ". " << resonance_legends[i] << std::endl;
        // }
        // if (resonance_legends.size() > 20)
        // {
        //     std::cout << "... and " << resonance_legends.size() - 20 << " more" << std::endl;
        // }

        // 测试定制legend功能
        std::cout << "\n=== Custom Legend Functions ===" << std::endl;

        // 测试所有定制legend
        std::cout << "\n--- All Custom Legends ---" << std::endl;
        const auto &custom_legends = parser.getCustomLegends();
        std::cout << "Total custom legends: " << custom_legends.size() << std::endl;
        for (size_t i = 0; i < custom_legends.size(); ++i)
        {
            std::cout << custom_legends[i] << std::endl;
        }

        // 测试按chain分类的定制legend
        // std::cout << "\n--- Custom Legends by Chain ---" << std::endl;
        // const auto &chain_custom_legends = parser.getChainCustomLegends();
        // for (const auto &[chain_name, legends] : chain_custom_legends)
        // {
        //     std::cout << "\nChain: " << chain_name << std::endl;
        //     std::cout << "Legend template: [";
        //     for (size_t i = 0; i < legends.size(); ++i)
        //     {
        //         std::cout << legends[i];
        //         if (i < legends.size() - 1)
        //             std::cout << ", ";
        //     }
        //     std::cout << "]" << std::endl;
        //     std::cout << "Generated legends: " << std::endl;
        //     for (size_t i = 0; i < legends.size(); ++i)
        //     {
        //         std::cout << legends[i] << std::endl;
        //     }
        // }

        std::cout << "\n=== Test Completed Successfully ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}