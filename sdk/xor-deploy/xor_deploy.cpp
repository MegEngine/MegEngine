#include <stdlib.h>
#include <iostream>
#include "megbrain/serialization/serializer.h"
using namespace mgb;

cg::ComputingGraph::OutputSpecItem make_callback_copy(SymbolVar dev,
                                                      HostTensorND& host) {
    auto cb = [&host](DeviceTensorND& d) { host.copy_from(d); };
    return {dev, cb};
}

int main(int argc, char* argv[]) {
    std::cout << " Usage: ./xornet_deploy model_name x_value y_value"
              << std::endl;
    if (argc != 4) {
        std::cout << " Wrong argument" << std::endl;
        return 0;
    }
    std::unique_ptr<serialization::InputFile> inp_file =
            serialization::InputFile::make_fs(argv[1]);
    float x = atof(argv[2]);
    float y = atof(argv[3]);
    auto loader = serialization::GraphLoader::make(std::move(inp_file));
    serialization::GraphLoadConfig config;
    serialization::GraphLoader::LoadResult network =
            loader->load(config, false);
    auto data = network.tensor_map["data"];
    float* data_ptr = data->resize({1, 2}).ptr<float>();
    data_ptr[0] = x;
    data_ptr[1] = y;
    HostTensorND predict;
    std::unique_ptr<cg::AsyncExecutable> func =
            network.graph->compile({make_callback_copy(
                    network.output_var_map.begin()->second, predict)});
    func->execute();
    func->wait();
    float* predict_ptr = predict.ptr<float>();
    std::cout << " Predicted: " << predict_ptr[0] << " " << predict_ptr[1]
              << std::endl;
}
