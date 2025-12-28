#include <Config.cuh>
#include <iostream>
#include <fstream>
#include <algorithm>

ConfigParser::ConfigParser(const std::string &config_file)
{
    try
    {
        YAML::Node config = YAML::LoadFile(config_file);

        if (config["Particles"])
            parseParticles(config["Particles"]);

        if (config["Data"])
            parseData(config["Data"]);

        if (config["DecayChains"])
            parseDecayChains(config["DecayChains"]);

        if (config["Resonances"])
            parseResonances(config["Resonances"]);

        if (config["Constraints"])
            parseConstraints(config["Constraints"]);

        if (config["Plot"])
            parsePlotConfig(config["Plot"]);
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "Error parsing config file: " << e.what() << std::endl;
        throw;
    }
}

std::vector<std::string> ConfigParser::getLegends() const
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

std::string ConfigParser::generateLegend(const std::vector<std::string> &particles) const
{
    std::string legend;
    for (size_t i = 0; i < particles.size(); ++i)
    {
        const auto &name = particles[i];

        // 检查是否是粒子
        auto particle_it = std::find_if(particles_.begin(), particles_.end(),
                                        [&](const Particle &p)
                                        { return p.name == name; });
        if (particle_it != particles_.end())
        {
            legend += particle_it->tex;
        }
        else
        {
            // 检查是否是共振态
            auto res_it = resonances_.find(name);
            if (res_it != resonances_.end())
                legend += res_it->second.tex;
            else
                legend += name; // 原始名称
        }

        if (i < particles.size() - 1)
            legend += " ";
    }
    return legend;
}

void ConfigParser::parseParticles(const YAML::Node &node)
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
        particle.tex = props["tex"].as<std::string>();

        particles_.push_back(particle);
    }
}

void ConfigParser::parseData(const YAML::Node &node)
{
    if (node["dat_order"])
        dat_order_ = node["dat_order"].as<std::vector<std::string>>();

    if (node["data"])
        data_files_["data"] = node["data"].as<std::vector<std::string>>();

    if (node["phsp"])
        data_files_["phsp"] = node["phsp"].as<std::vector<std::string>>();

    if (node["bkg"])
        data_files_["bkg"] = node["bkg"].as<std::vector<std::string>>();
}

void ConfigParser::parseDecayChains(const YAML::Node &node)
{
    for (const auto &chain_node : node)
    {
        std::string chain_name = chain_node.first.as<std::string>();
        const auto &chain_data = chain_node.second;

        DecayChainConfig chain;
        chain.name = chain_name;

        // 解析衰变步骤
        if (chain_data["decay"])
        {
            for (const auto &step_node : chain_data["decay"])
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

        // 解析共振态链
        for (const auto &res_node : chain_data)
        {
            std::string key = res_node.first.as<std::string>();
            if (key != "decay" && key != "legend")
            {
                ResonanceChainConfig res_chain;
                res_chain.intermediate = key;

                for (const auto &spin_node : res_node.second)
                {
                    for (const auto &spin_pair : spin_node)
                    {
                        SpinChainConfig spin_chain;

                        // 解析自旋宇称 [J, P]
                        if (spin_pair.first.IsSequence())
                        {
                            for (const auto &jp : spin_pair.first)
                            {
                                if (jp["J"])
                                    spin_chain.spin_parity.push_back(jp["J"].as<int>());
                                if (jp["P"])
                                    spin_chain.spin_parity.push_back(jp["P"].as<int>());
                            }
                        }

                        // 解析共振态列表
                        spin_chain.resonances = spin_pair.second.as<std::vector<std::string>>();
                        res_chain.spin_chains.push_back(spin_chain);
                    }
                }
                chain.resonance_chains.push_back(res_chain);
            }
        }

        // 解析legend模板
        if (chain_data["legend"])
            chain.legend_template = chain_data["legend"].as<std::vector<std::string>>();

        decay_chains_.push_back(chain);
    }
}

void ConfigParser::parseResonances(const YAML::Node &node)
{
    for (const auto &res_node : node)
    {
        std::string name = res_node.first.as<std::string>();
        const auto &props = res_node.second;

        ResonanceConfig res;
        res.name = name;
        res.J = props["J"].as<int>();
        res.P = props["P"].as<int>();
        res.type = props["model"].as<std::string>();
        res.parameters = props["parameters"].as<std::vector<double>>();
        res.tex = props["tex"].as<std::string>();

        resonances_[name] = res;
    }
}

void ConfigParser::parseConstraints(const YAML::Node &node)
{
    constraints_.clear();

    // 解析 full 约束（同时包含实部和虚部）
    if (node["trans"])
    {
        for (const auto &constraint_list : node["trans"])
        {
            for (const auto &pair : constraint_list)
            {
                ConstraintConfig constraint;
                constraint.type = "trans";

                // 解析链名列表
                constraint.names = pair.first.as<std::vector<std::string>>();

                // 解析约束值
                const YAML::Node &values = pair.second;
                if (values.IsSequence())
                {
                    for (const auto &row : values)
                    {
                        if (row.IsSequence() && row.size() >= 2)
                        {
                            double real_val = row[0].as<double>();
                            double imag_val = row[1].as<double>();
                            // constraint.constraints.emplace_back(real_val, imag_val);
                            constraint.values.push_back(std::complex<double>(real_val, imag_val));
                        }
                    }
                }

                constraints_.push_back(constraint);
            }
        }
    }
}

void ConfigParser::parsePlotConfig(const YAML::Node &node)
{
    plot_configs_.clear();

    // 解析mass图配置
    if (node["mass"])
    {
        for (const auto &plot_item : node["mass"])
        {
            PlotConfig config;
            config.type = "mass";

            // 解析particles
            if (plot_item["input"])
            {
                const YAML::Node &particles_node = plot_item["input"];
                if (particles_node.IsSequence())
                {
                    // 检查是否是一维序列
                    bool is_2d = false;
                    for (const auto &elem : particles_node)
                    {
                        if (elem.IsSequence())
                        {
                            is_2d = true;
                            break;
                        }
                    }
                    if (is_2d)
                    {
                        // 二维序列
                        for (const auto &group : particles_node)
                        {
                            config.particles.push_back(group.as<std::vector<std::string>>());
                        }
                    }
                    else
                    {
                        // 一维序列，包装成二维
                        config.particles.push_back(particles_node.as<std::vector<std::string>>());
                    }
                }
            }

            // bins: 单个整数
            config.bins = {plot_item["bins"].as<int>()};

            // range: 一维数组，转换为二维数组
            std::vector<double> range = plot_item["range"].as<std::vector<double>>();
            config.ranges = {range};

            // display: 一维数组（两个字符串）
            config.display = plot_item["display"].as<std::vector<std::string>>();

            plot_configs_.push_back(config);
        }
    }

    // 解析cosbeta图配置
    if (node["cosbeta"])
    {
        for (const auto &plot_item : node["cosbeta"])
        {
            PlotConfig config;
            config.type = "cosbeta";

            // 解析particles（二维列表）
            if (plot_item["input"])
            {
                const YAML::Node &particles_node = plot_item["input"];
                for (const auto &group : particles_node)
                {
                    config.particles.push_back(group.as<std::vector<std::string>>());
                }
            }

            // bins: 单个整数
            config.bins = {plot_item["bins"].as<int>()};

            // range: 一维数组，转换为二维数组
            std::vector<double> range = plot_item["range"].as<std::vector<double>>();
            config.ranges = {range};

            // display: 一维数组（两个字符串）
            config.display = plot_item["display"].as<std::vector<std::string>>();

            plot_configs_.push_back(config);
        }
    }

    // 解析dalitz图配置
    if (node["dalitz"])
    {
        for (const auto &plot_item : node["dalitz"])
        {
            PlotConfig config;
            config.type = "dalitz";

            // 解析particles（二维列表，包含两个粒子组）
            if (plot_item["input"])
            {
                const YAML::Node &particles_node = plot_item["input"];
                for (const auto &group : particles_node)
                {
                    config.particles.push_back(group.as<std::vector<std::string>>());
                }
            }

            // bins: 二维数组（两个整数）
            config.bins = plot_item["bins"].as<std::vector<int>>();

            // range: 二维数组
            const YAML::Node &range_node = plot_item["range"];
            for (const auto &range : range_node)
            {
                config.ranges.push_back(range.as<std::vector<double>>());
            }

            // display: 一维数组（两个字符串）
            config.display = plot_item["display"].as<std::vector<std::string>>();

            plot_configs_.push_back(config);
        }
    }
}

// ConfigParser::ConfigParser(const std::string &config_file)
// {
//     try
//     {
//         YAML::Node config = YAML::LoadFile(config_file);

//         if (config["Particles"])
//         {
//             parseParticles(config["Particles"]);
//         }

//         if (config["Data"])
//         {
//             parseData(config["Data"]);
//         }

//         if (config["DecayChains"])
//         {
//             parseDecayChains(config["DecayChains"]);
//         }

//         if (config["Resonances"])
//         {
//             parseResonances(config["Resonances"]);
//         }

//         if (config["conjugate_pairs"])
//         {
//             parseConjugatePairs(config["conjugate_pairs"]);
//         }

//         if (config["Plot"])
//         {
//             parsePlotConfig(config["Plot"]);
//         }
//     }
//     catch (const YAML::Exception &e)
//     {
//         std::cerr << "Error parsing config file: " << e.what() << std::endl;
//         throw;
//     }
// }

// std::vector<std::string> ConfigParser::getCustomLegends() const
// {
//     return generateCustomLegends();
// }

// std::map<std::string, std::vector<std::string>> ConfigParser::getChainCustomLegends() const
// {
//     return generateChainCustomLegends();
// }

// std::string ConfigParser::generateLegend(const std::vector<std::string> &particles) const
// {
//     std::string legend;
//     for (size_t i = 0; i < particles.size(); ++i)
//     {
//         const auto &particle_name = particles[i];

//         // 首先检查是否是粒子
//         auto particle_it = std::find_if(particles_.begin(), particles_.end(),
//                                         [&](const Particle &p)
//                                         { return p.name == particle_name; });
//         if (particle_it != particles_.end())
//         {
//             legend += particle_it->tex;
//         }
//         else
//         {
//             // 如果不是粒子，检查是否是共振态
//             auto res_it = resonances_.find(particle_name);
//             if (res_it != resonances_.end())
//             {
//                 legend += res_it->second.tex;
//             }
//             else
//             {
//                 // 如果都不是，使用原始名称作为字符输出
//                 legend += particle_name;
//             }
//         }

//         if (i < particles.size() - 1)
//         {
//             legend += "";
//         }
//     }
//     return legend;
// }

// std::string ConfigParser::getParticleTex(const std::string &name) const
// {
//     auto particle_it = std::find_if(particles_.begin(), particles_.end(),
//                                     [&](const Particle &p)
//                                     { return p.name == name; });
//     if (particle_it != particles_.end())
//     {
//         return particle_it->tex;
//     }
//     return name; // 如果找不到，返回原始名称
// }

// std::string ConfigParser::getResonanceTex(const std::string &name) const
// {
//     auto res_it = resonances_.find(name);
//     if (res_it != resonances_.end())
//     {
//         return res_it->second.tex;
//     }
//     return name; // 如果找不到，返回原始名称
// }

// std::vector<std::string> ConfigParser::getAllLegends() const
// {
//     std::vector<std::string> legends;

//     for (const auto &chain : decay_chains_)
//     {
//         // 为每个decay chain生成legend
//         for (const auto &step : chain.decay_steps)
//         {
//             // 生成母粒子和子粒子的legend
//             std::vector<std::string> particles_for_legend;
//             particles_for_legend.push_back(step.mother);
//             particles_for_legend.insert(particles_for_legend.end(),
//                                         step.daughters.begin(), step.daughters.end());

//             std::string legend = generateLegend(particles_for_legend);
//             legends.push_back(legend);
//         }

//         // 为每个共振态链生成legend
//         for (const auto &res_chain : chain.resonance_chains)
//         {
//             for (const auto &spin_chain : res_chain.spin_chains)
//             {
//                 for (const auto &resonance : spin_chain.resonances)
//                 {
//                     // 生成中间态和共振态的legend
//                     std::vector<std::string> particles_for_legend;
//                     particles_for_legend.push_back(res_chain.intermediate);
//                     particles_for_legend.push_back(resonance);

//                     std::string legend = generateLegend(particles_for_legend);
//                     legends.push_back(legend);
//                 }
//             }
//         }
//     }

//     return legends;
// }

// std::vector<std::string> ConfigParser::getAllResonanceCombinationLegends() const
// {
//     std::vector<std::string> legends;

//     for (const auto &chain : decay_chains_)
//     {
//         // 构建中间态到共振态的映射
//         std::map<std::pair<std::string, std::vector<int>>, std::vector<std::string>> spin_resonance_map;
//         std::vector<std::vector<Particle>> particleLists;

//         for (const auto &res_chain : chain.resonance_chains)
//         {
//             std::vector<Particle> particles;
//             for (const auto &spin_chain : res_chain.spin_chains)
//             {
//                 std::pair<std::string, std::vector<int>> key = {res_chain.intermediate,
//                                                                 {spin_chain.spin_parity[0], spin_chain.spin_parity[1]}};
//                 spin_resonance_map[key] = spin_chain.resonances;
//                 particles.push_back({res_chain.intermediate, spin_chain.spin_parity[0],
//                                      spin_chain.spin_parity[1], -1.0, ""});
//             }
//             particleLists.push_back(particles);
//         }

//         // 生成所有JP组合
//         std::vector<std::vector<Particle>> result = {{}};
//         for (const auto &particleList : particleLists)
//         {
//             std::vector<std::vector<Particle>> temp;
//             for (const auto &res : result)
//             {
//                 for (const auto &particle : particleList)
//                 {
//                     std::vector<Particle> new_res = res;
//                     new_res.push_back(particle);
//                     temp.push_back(new_res);
//                 }
//             }
//             result = std::move(temp);
//         }

//         // 为每个JP组合生成共振态组合的legend
//         for (size_t i = 0; i < result.size(); ++i)
//         {
//             const auto &jp_combination = result[i];

//             // 为当前JP组合生成所有共振态组合
//             std::vector<std::vector<std::pair<std::string, std::string>>> resonance_combinations = {{}};

//             for (const auto &particle : jp_combination)
//             {
//                 std::pair<std::string, std::vector<int>> key = {particle.name, {particle.spin, particle.parity}};
//                 const auto &resonances = spin_resonance_map[key];

//                 std::vector<std::vector<std::pair<std::string, std::string>>> temp_res;
//                 for (const auto &current_combo : resonance_combinations)
//                 {
//                     for (const auto &resonance : resonances)
//                     {
//                         std::vector<std::pair<std::string, std::string>> new_combo = current_combo;
//                         new_combo.push_back({particle.name, resonance});
//                         temp_res.push_back(new_combo);
//                     }
//                 }
//                 resonance_combinations = std::move(temp_res);
//             }

//             // 为每个共振态组合生成legend
//             for (const auto &res_combo : resonance_combinations)
//             {
//                 std::vector<std::string> particles_for_legend;
//                 for (const auto &res_pair : res_combo)
//                 {
//                     particles_for_legend.push_back(res_pair.second);
//                 }
//                 std::string legend = generateLegend(particles_for_legend);
//                 legends.push_back(legend);
//             }
//         }
//     }

//     return legends;
// }

// std::vector<std::string> ConfigParser::generateCustomLegends() const
// {
//     std::vector<std::string> legends;

//     for (const auto &chain : decay_chains_)
//     {
//         if (!chain.legend_template.empty())
//         {
//             // 检查legend模板中的占位符
//             std::map<std::string, std::vector<std::string>> placeholder_map;
//             std::vector<std::string> template_items;

//             for (const auto &item : chain.legend_template)
//             {
//                 // 检查是否是中间态占位符（以R_开头）
//                 if (item.find("R_") == 0)
//                 {
//                     // 如果这个占位符还没有处理过，找到对应的共振态链
//                     if (placeholder_map.find(item) == placeholder_map.end())
//                     {
//                         std::vector<std::string> resonances;
//                         for (const auto &res_chain : chain.resonance_chains)
//                         {
//                             if (res_chain.intermediate == item)
//                             {
//                                 for (const auto &spin_chain : res_chain.spin_chains)
//                                 {
//                                     resonances.insert(resonances.end(),
//                                                       spin_chain.resonances.begin(),
//                                                       spin_chain.resonances.end());
//                                 }
//                                 break;
//                             }
//                         }
//                         placeholder_map[item] = resonances;
//                     }
//                 }
//                 template_items.push_back(item);
//             }

//             // 生成所有组合
//             if (!placeholder_map.empty())
//             {
//                 // 获取唯一的占位符列表
//                 std::vector<std::string> unique_placeholders;
//                 for (const auto &[placeholder, resonances] : placeholder_map)
//                 {
//                     unique_placeholders.push_back(placeholder);
//                 }

//                 // 为每个唯一占位符生成共振态组合
//                 std::vector<std::vector<std::string>> combinations = {{}};
//                 for (const auto &placeholder : unique_placeholders)
//                 {
//                     const auto &resonance_list = placeholder_map[placeholder];
//                     std::vector<std::vector<std::string>> temp;
//                     for (const auto &current_combo : combinations)
//                     {
//                         for (const auto &resonance : resonance_list)
//                         {
//                             std::vector<std::string> new_combo = current_combo;
//                             new_combo.push_back(resonance);
//                             temp.push_back(new_combo);
//                         }
//                     }
//                     combinations = std::move(temp);
//                 }

//                 // 为每个组合生成legend
//                 for (const auto &combo : combinations)
//                 {
//                     std::vector<std::string> particles_for_legend;

//                     // 创建占位符到共振态的映射
//                     std::map<std::string, std::string> placeholder_to_resonance;
//                     for (size_t i = 0; i < unique_placeholders.size(); ++i)
//                     {
//                         placeholder_to_resonance[unique_placeholders[i]] = combo[i];
//                     }

//                     // 构建legend
//                     for (const auto &item : template_items)
//                     {
//                         if (item.find("R_") == 0)
//                         {
//                             particles_for_legend.push_back(placeholder_to_resonance[item]);
//                         }
//                         else
//                         {
//                             particles_for_legend.push_back(item);
//                         }
//                     }

//                     std::string legend = generateLegend(particles_for_legend);
//                     legends.push_back(legend);
//                 }
//             }
//             else
//             {
//                 // 如果没有占位符，直接生成legend
//                 std::string legend = generateLegend(chain.legend_template);
//                 legends.push_back(legend);
//             }
//         }
//     }

//     return legends;
// }

// std::map<std::string, std::vector<std::string>> ConfigParser::generateChainCustomLegends() const
// {
//     std::map<std::string, std::vector<std::string>> chain_legends;

//     for (const auto &chain : decay_chains_)
//     {
//         if (!chain.legend_template.empty())
//         {
//             std::vector<std::string> legends;

//             // 检查legend模板中的占位符
//             std::map<std::string, std::vector<std::string>> placeholder_map;
//             std::vector<std::string> template_items;

//             for (const auto &item : chain.legend_template)
//             {
//                 // 检查是否是中间态占位符（以R_开头）
//                 if (item.find("R_") == 0)
//                 {
//                     // 如果这个占位符还没有处理过，找到对应的共振态链
//                     if (placeholder_map.find(item) == placeholder_map.end())
//                     {
//                         std::vector<std::string> resonances;
//                         for (const auto &res_chain : chain.resonance_chains)
//                         {
//                             if (res_chain.intermediate == item)
//                             {
//                                 for (const auto &spin_chain : res_chain.spin_chains)
//                                 {
//                                     resonances.insert(resonances.end(),
//                                                       spin_chain.resonances.begin(),
//                                                       spin_chain.resonances.end());
//                                 }
//                                 break;
//                             }
//                         }
//                         placeholder_map[item] = resonances;
//                     }
//                 }
//                 template_items.push_back(item);
//             }

//             // 生成所有组合
//             if (!placeholder_map.empty())
//             {
//                 // 获取唯一的占位符列表
//                 std::vector<std::string> unique_placeholders;
//                 for (const auto &[placeholder, resonances] : placeholder_map)
//                 {
//                     unique_placeholders.push_back(placeholder);
//                 }

//                 // 为每个唯一占位符生成共振态组合
//                 std::vector<std::vector<std::string>> combinations = {{}};
//                 for (const auto &placeholder : unique_placeholders)
//                 {
//                     const auto &resonance_list = placeholder_map[placeholder];
//                     std::vector<std::vector<std::string>> temp;
//                     for (const auto &current_combo : combinations)
//                     {
//                         for (const auto &resonance : resonance_list)
//                         {
//                             std::vector<std::string> new_combo = current_combo;
//                             new_combo.push_back(resonance);
//                             temp.push_back(new_combo);
//                         }
//                     }
//                     combinations = std::move(temp);
//                 }

//                 // 为每个组合生成legend
//                 for (const auto &combo : combinations)
//                 {
//                     std::vector<std::string> particles_for_legend;

//                     // 创建占位符到共振态的映射
//                     std::map<std::string, std::string> placeholder_to_resonance;
//                     for (size_t i = 0; i < unique_placeholders.size(); ++i)
//                     {
//                         placeholder_to_resonance[unique_placeholders[i]] = combo[i];
//                     }

//                     // 构建legend
//                     for (const auto &item : template_items)
//                     {
//                         if (item.find("R_") == 0)
//                         {
//                             particles_for_legend.push_back(placeholder_to_resonance[item]);
//                         }
//                         else
//                         {
//                             particles_for_legend.push_back(item);
//                         }
//                     }

//                     std::string legend = generateLegend(particles_for_legend);
//                     legends.push_back(legend);
//                 }
//             }
//             else
//             {
//                 // 如果没有占位符，直接生成legend
//                 std::string legend = generateLegend(chain.legend_template);
//                 legends.push_back(legend);
//             }

//             chain_legends[chain.name] = legends;
//         }
//     }

//     return chain_legends;
// }

// void ConfigParser::parseParticles(const YAML::Node &node)
// {
//     for (const auto &particle_node : node)
//     {
//         std::string name = particle_node.first.as<std::string>();
//         const auto &props = particle_node.second;

//         Particle particle;
//         particle.name = name;
//         particle.spin = props["J"].as<int>();
//         particle.parity = props["P"].as<int>();
//         particle.mass = props["mass"].as<double>();
//         if (props["tex"])
//         {
//             particle.tex = props["tex"].as<std::string>();
//         }
//         else
//         {
//             particle.tex = name; // 如果没有tex字段，使用name作为默认值
//         }

//         particles_.push_back(particle);
//     }
// }

// void ConfigParser::parseData(const YAML::Node &node)
// {
//     if (node["dat_order"])
//     {
//         dat_order_ = node["dat_order"].as<std::vector<std::string>>();
//     }

//     if (node["data"])
//     {
//         data_files_["data"] = node["data"].as<std::vector<std::string>>();
//     }

//     if (node["phsp"])
//     {
//         data_files_["phsp"] = node["phsp"].as<std::vector<std::string>>();
//     }

//     if (node["bkg"])
//     {
//         data_files_["bkg"] = node["bkg"].as<std::vector<std::string>>();
//     }
// }

// void ConfigParser::parseDecayChains(const YAML::Node &node)
// {
//     for (const auto &chain_node : node)
//     {
//         std::string chain_name = chain_node.first.as<std::string>();

//         DecayChainConfig chain;
//         chain.name = chain_name;

//         // 解析衰变步骤
//         if (chain_node.second["decay"])
//         {
//             const auto &decay_steps = chain_node.second["decay"];
//             for (const auto &step_node : decay_steps)
//             {
//                 // 解析 {mother, [daughter1, daughter2, ...]} 格式
//                 if (step_node.IsMap())
//                 {
//                     for (const auto &decay_pair : step_node)
//                     {
//                         DecayStep step;
//                         step.mother = decay_pair.first.as<std::string>();
//                         step.daughters = decay_pair.second.as<std::vector<std::string>>();
//                         chain.decay_steps.push_back(step);
//                     }
//                 }
//             }
//         }

//         // 解析共振态链配置
//         for (const auto &res_chain_node : chain_node.second)
//         {
//             std::string key = res_chain_node.first.as<std::string>();
//             if (key != "decay" && key != "legend")
//             {
//                 ResonanceChainConfig res_chain;
//                 res_chain.intermediate = key;

//                 for (const auto &spin_config : res_chain_node.second)
//                 {
//                     // Parse new format: [{J: 1}, {P: -1}]: [resonances...]
//                     // spin_config is a sequence of maps
//                     for (const auto &spin_pair : spin_config)
//                     {
//                         SpinChainConfig spin_chain;

//                         // The key is actually a sequence: [{J: 1}, {P: -1}]
//                         // We need to extract J and P from this sequence
//                         std::vector<int> spin_parity;

//                         if (spin_pair.first.IsSequence())
//                         {
//                             for (const auto &j_p_node : spin_pair.first)
//                             {
//                                 if (j_p_node.IsMap())
//                                 {
//                                     if (j_p_node["J"])
//                                     {
//                                         spin_parity.push_back(j_p_node["J"].as<int>());
//                                     }
//                                     if (j_p_node["P"])
//                                     {
//                                         spin_parity.push_back(j_p_node["P"].as<int>());
//                                     }
//                                 }
//                             }
//                         }
//                         else
//                         {
//                             // Fallback: try to parse as vector (old format)
//                             spin_parity = spin_pair.first.as<std::vector<int>>();
//                         }

//                         spin_chain.spin_parity = spin_parity;
//                         spin_chain.resonances = spin_pair.second.as<std::vector<std::string>>();
//                         res_chain.spin_chains.push_back(spin_chain);
//                     }
//                 }

//                 chain.resonance_chains.push_back(res_chain);
//             }
//         }

//         // 解析legend模板
//         if (chain_node.second["legend"])
//         {
//             chain.legend_template = chain_node.second["legend"].as<std::vector<std::string>>();
//         }

//         decay_chains_.push_back(chain);
//     }
// }

// void ConfigParser::parseResonances(const YAML::Node &node)
// {
//     for (const auto &res_node : node)
//     {
//         std::string name = res_node.first.as<std::string>();
//         const auto &props = res_node.second;

//         ResonanceConfig res;
//         res.name = name;
//         if (props["J"])
//         {
//             res.J = props["J"].as<int>();
//         }
//         else
//         {
//             res.J = -1; // 如果没有J字段，设为-1
//         }
//         if (props["P"])
//         {
//             res.P = props["P"].as<int>();
//         }
//         else
//         {
//             res.P = 0; // 如果没有P字段，设为0
//         }
//         res.type = props["model"].as<std::string>();
//         res.parameters = props["parameters"].as<std::vector<double>>();
//         if (props["tex"])
//         {
//             res.tex = props["tex"].as<std::string>();
//         }
//         else
//         {
//             res.tex = name; // 如果没有tex字段，使用name作为默认值
//         }

//         resonances_[name] = res;
//     }
// }

// void ConfigParser::parseConjugatePairs(const YAML::Node &node)
// {
//     for (const auto &pair_node : node)
//     {
//         auto pair = pair_node.as<std::vector<std::string>>();
//         if (pair.size() == 2)
//         {
//             conjugate_pairs_.emplace_back(pair[0], pair[1]);
//         }
//     }
// }

// void ConfigParser::parseConstraints(const YAML::Node &node)
// {
//     for (const auto &constraint_node : node)
//     {
//         ConstraintConfig constraint;
//         constraint.names = constraint_node["names"].as<std::vector<std::string>>();
//         constraint.limits = constraint_node["limits"].as<std::vector<std::pair<double, double>>>();
//         constraints_.push_back(constraint);
//     }
// }

// void ConfigParser::parsePlots(const YAML::Node &node)
// {
//     for (const auto &plot_node : node)
//     {
//         std::string plot_type = plot_node.first.as<std::string>();

//         // 根据plot_type进行相应的处理
//         if (plot_type == "mass")
//         {
//             for (const auto &plot_item : node["mass"])
//             {
//                 PlotConfig::MassPlot mass_plot;

//                 // 解析 particles
//                 if (plot_item[0].IsSequence())
//                 {
//                     mass_plot.particles = plot_item[0].as<std::vector<std::string>>();
//                 }

//                 // 解析 bins
//                 if (plot_item[1].IsScalar())
//                 {
//                     mass_plot.bins = plot_item[1].as<int>();
//                 }

//                 // 解析 range
//                 if (plot_item[2].IsSequence() && plot_item[2].size() >= 2)
//                 {
//                     mass_plot.range = plot_item[2].as<std::vector<double>>();
//                 }

//                 // 解析 tex
//                 if (plot_item[3].IsSequence() && plot_item[3].size() >= 2)
//                 {
//                     mass_plot.tex = plot_item[3].as<std::vector<std::string>>();
//                 }

//                 if (!mass_plot.particles.empty() && mass_plot.bins > 0 &&
//                     mass_plot.range.size() >= 2)
//                 {
//                     plot_config_.mass_plots.push_back(mass_plot);
//                 }
//                 else
//                 {
//                     std::cerr << "Warning: Invalid mass plot configuration ignored." << std::endl;
//                 }
//             }
//         }
//         else if (plot_type == "cosbeta")
//         {
//             // 处理角度直方图配置
//         }
//         else if (plot_type == "dalitz")
//         {
//             // 处理Dalitz直方图配置
//         }
//     }
// }