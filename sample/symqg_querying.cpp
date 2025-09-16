#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include <gflags/gflags.h>

#include "defines.hpp"
#include "index/symqg/qg.hpp"
#include "utils/io.hpp"
#include "utils/BS_thread_pool.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::symqg::QuantizedGraph<float>;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

DEFINE_int32(num_threads, 1, "Number of threads to use");
DEFINE_string(output_csv_path, "", "Output CSV file path");

// std::vector<size_t> efs = {
//     10, 20, 40, 50, 60, 80, 100, 150, 170, 190, 200, 250, 300, 400, 500, 600, 700, 800, 1500
// };
std::vector<size_t> efs = {400};

size_t test_round = 3;
size_t topk = 10;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3>\n"
                  << "arg1: path for index \n"
                  << "arg2: path for query file, format .fvecs\n"
                  << "arg3: path for groundtruth file format .ivecs\n";
        exit(1);
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    char* index_file = argv[1];
    char* query_file = argv[2];
    char* gt_file = argv[3];

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    size_t nq = query.rows();
    size_t total_count = nq * topk;

    index_type qg;
    qg.load(index_file);

    auto run_q_f = [&](int num_threads){
        using rabitqlib::symqg::QueryRuntimeMetrics;
        std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(efs.size()));
        std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(efs.size()));
        std::vector<std::vector<float>> all_compute_kopps(test_round, std::vector<float>(efs.size()));
        std::vector<std::vector<float>> all_bw_mbps(test_round, std::vector<float>(efs.size()));

        for (size_t i = 0; i < efs.size(); ++i) {
            for (size_t r = 0; r < test_round; r++) {
                size_t ef = efs[i];
                qg.set_ef(ef);

                // Pre-allocate per-query results to avoid contention; each query writes to its own slot.
                std::vector<std::vector<PID>> results_per_query(nq, std::vector<PID>(topk));

                BS::light_thread_pool pool(static_cast<std::size_t>(num_threads));
                std::vector<QueryRuntimeMetrics> metrics(nq);

                auto start = std::chrono::high_resolution_clock::now();

                pool.detach_loop<std::size_t, std::size_t>(0, nq, [&](std::size_t z) {
                    QueryRuntimeMetrics m;
                    qg.search(&query(z, 0), topk, results_per_query[z].data(), m);
                    metrics[z] = m;
                });
                pool.wait();

                auto end = std::chrono::high_resolution_clock::now();
                float elapsed_us = std::chrono::duration<float, std::micro>(end - start).count();

                float qps = static_cast<float>(nq) / (elapsed_us / 1e6F);
                QueryRuntimeMetrics metrics_sum;
                // Compute recall after timing.
                size_t total_correct = 0;
                for (size_t z = 0; z < nq; z++) {
                    for (size_t y = 0; y < topk; y++) {
                        for (size_t k = 0; k < topk; k++) {
                            if (gt(z, k) == results_per_query[z][y]) {
                                total_correct++;
                                break;
                            }
                        }
                    }
                    // std::cout << "metrics[" << z << "].total_comp_cnt: " << metrics[z].total_comp_cnt << std::endl;
                    metrics_sum.fast_bitsum += metrics[z].fast_bitsum;
                    metrics_sum.acc_bitsum += metrics[z].acc_bitsum;
                    metrics_sum.total_comp_cnt += metrics[z].total_comp_cnt;
                }
                float recall = static_cast<float>(total_correct) / static_cast<float>(total_count);

                all_qps[r][i] = qps;
                all_recall[r][i] = recall;
                all_bw_mbps[r][i] = (metrics_sum.fast_bitsum + metrics_sum.acc_bitsum) / 8.0 / (1<<20) / (elapsed_us / 1e6F);
                all_compute_kopps[r][i] = metrics_sum.total_comp_cnt / 1000.0 / (elapsed_us / 1e6F);
            }
        }

        auto avg_qps = rabitqlib::horizontal_avg(all_qps);
        auto avg_recall = rabitqlib::horizontal_avg(all_recall);
        auto avg_compute_kopps = rabitqlib::horizontal_avg(all_compute_kopps);
        auto avg_bw_mbps = rabitqlib::horizontal_avg(all_bw_mbps);

        std::ostringstream oss;
        for (size_t i = 0; i < avg_qps.size(); ++i) {
            oss << efs[i] << "," << num_threads << "," << avg_qps[i] << "," << avg_recall[i] << "," <<  avg_compute_kopps[i] << "," << avg_bw_mbps[i] << "," << '\n';
        }
        const std::string output_text = oss.str();
        std::cout << output_text;
        return output_text;
    };

    std::string output_text = "EF,num_threads,QPS,Recall,compute_kopps,bw_mbps,\n";
    std::cout << output_text;

    if (FLAGS_num_threads) {
        output_text += run_q_f(FLAGS_num_threads);
    } else {
        output_text += run_q_f(1);
        output_text += run_q_f(2);
        output_text += run_q_f(4);
        output_text += run_q_f(6);
        output_text += run_q_f(8);
        output_text += run_q_f(10);
        output_text += run_q_f(12);
        output_text += run_q_f(16);
        output_text += run_q_f(20);
        output_text += run_q_f(24);
        output_text += run_q_f(28);
        output_text += run_q_f(32);
        output_text += run_q_f(40);
        output_text += run_q_f(48);
    }

    if (!FLAGS_output_csv_path.empty()) {
        std::ofstream ofs(FLAGS_output_csv_path);
        if (ofs.is_open()) {
            ofs << output_text;
            ofs.close();
            std::cout << "Output CSV file: " << FLAGS_output_csv_path << "\n";
        } else {
            std::cerr << "Failed to open output CSV file: " << FLAGS_output_csv_path << "\n";
        }
    }

    return 0;
}