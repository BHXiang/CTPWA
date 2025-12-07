#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>

int main() {
    try {
        YAML::Node config = YAML::LoadFile("config.yml");

        std::cout << "=== Debug Config Structure ===" << std::endl;

        if (config["decay_chains"]) {
            std::cout << "\n=== Decay Chains ===" << std::endl;
            for (const auto &chain_node : config["decay_chains"]) {
                std::string chain_name = chain_node.first.as<std::string>();
                std::cout << "Chain: " << chain_name << std::endl;

                for (const auto &res_chain_node : chain_node.second) {
                    std::string key = res_chain_node.first.as<std::string>();
                    std::cout << "  Key: " << key << " (type: " << res_chain_node.second.Type() << ")" << std::endl;

                    if (key != "decay") {
                        std::cout << "  Intermediate: " << key << std::endl;
                        for (const auto &spin_config : res_chain_node.second) {
                            std::cout << "    Spin config type: " << spin_config.Type() << std::endl;
                            for (const auto &spin_pair : spin_config) {
                                std::cout << "      Key type: " << spin_pair.first.Type() << std::endl;
                                std::cout << "      Key: " << spin_pair.first << std::endl;
                                std::cout << "      Value type: " << spin_pair.second.Type() << std::endl;
                                std::cout << "      Value: " << spin_pair.second << std::endl;
                            }
                        }
                    }
                }
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}