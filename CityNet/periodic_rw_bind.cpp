#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>
#include <stdexcept>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

py::array_t<uint32_t> random_walks_periodic_restart_no_backtrack(
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> _indptr,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> _indices,
    py::array_t<float,    py::array::c_style | py::array::forcecast> _data,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> _startNodes,
    size_t seed,
    size_t nWalks,
    size_t walkLen,
    size_t period)
{
    auto indptrBuf = _indptr.request();
    auto indicesBuf = _indices.request();
    auto dataBuf = _data.request();
    auto startNodesBuf = _startNodes.request();

    if (indptrBuf.ndim != 1 || indicesBuf.ndim != 1 || dataBuf.ndim != 1 || startNodesBuf.ndim != 1) {
        throw std::runtime_error("All inputs must be 1D arrays.");
    }

    auto *indptr = static_cast<uint32_t *>(indptrBuf.ptr);
    auto *indices = static_cast<uint32_t *>(indicesBuf.ptr);
    auto *data = static_cast<float *>(dataBuf.ptr);
    auto *startNodes = static_cast<uint32_t *>(startNodesBuf.ptr);

    size_t nNodes = static_cast<size_t>(startNodesBuf.shape[0]);
    size_t shape = nWalks * nNodes;

    py::array_t<uint32_t> _walks({static_cast<py::ssize_t>(shape),
                                  static_cast<py::ssize_t>(walkLen)});
    auto walksBuf = _walks.request();
    auto *walks = static_cast<uint32_t *>(walksBuf.ptr);

    #pragma omp parallel for if(shape > 256)
    for (long long i = 0; i < static_cast<long long>(shape); i++) {
        size_t thread_seed = seed + static_cast<size_t>(i);
        std::mt19937 generator(static_cast<uint32_t>(thread_seed));
        std::uniform_real_distribution<float> dist01(0.f, 1.f);

        std::vector<float> draws(walkLen > 0 ? walkLen - 1 : 0);
        for (size_t z = 0; z + 1 < walkLen; z++) {
            draws[z] = dist01(generator);
        }

        uint32_t step = startNodes[static_cast<size_t>(i) % nNodes];
        uint32_t startNode = step;
        walks[static_cast<size_t>(i) * walkLen + 0] = step;

        for (size_t k = 1; k < walkLen; k++) {
            if (period > 0 && k % period == 0) {
                step = startNode;
                walks[static_cast<size_t>(i) * walkLen + k] = step;
                continue;
            }

            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            if (start == end) {
                walks[static_cast<size_t>(i) * walkLen + k] = step;
                continue;
            }

            uint32_t prev = UINT32_MAX;
            bool use_no_backtrack = (k >= 2);
            if (use_no_backtrack) {
                prev = walks[static_cast<size_t>(i) * walkLen + (k - 2)];
            }

            float weightSum = 0.f;
            std::vector<float> weights(static_cast<size_t>(end - start), 0.f);

            for (uint32_t z = start; z < end; z++) {
                uint32_t neighbor = indices[z];
                float weight = data[z];

                if (use_no_backtrack && neighbor == prev) {
                    weight = 0.f;
                }

                weights[static_cast<size_t>(z - start)] = weight;
                weightSum += weight;
            }

            // fallback: if all allowed weights vanished, allow ordinary transition
            if (weightSum <= 0.f) {
                for (uint32_t z = start; z < end; z++) {
                    weights[static_cast<size_t>(z - start)] = data[z];
                    weightSum += data[z];
                }
            }

            float draw = draws[k - 1] * weightSum;
            float cumsum = 0.f;
            size_t index = 0;
            for (size_t z = 0; z < static_cast<size_t>(end - start); z++) {
                cumsum += weights[z];
                if (draw <= cumsum) {
                    index = z;
                    break;
                }
            }

            step = indices[start + static_cast<uint32_t>(index)];
            walks[static_cast<size_t>(i) * walkLen + k] = step;
        }
    }

    return _walks;
}

PYBIND11_MODULE(periodic_rw_ext, m) {
    m.doc() = "Periodic-restart no-backtrack random walks";

    m.def(
        "random_walks_periodic_restart_no_backtrack",
        &random_walks_periodic_restart_no_backtrack,
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("data"),
        py::arg("start_nodes"),
        py::arg("seed"),
        py::arg("n_walks"),
        py::arg("walk_len"),
        py::arg("period")
    );
}