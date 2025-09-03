#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include <gflags/gflags.h>

#include "index/hnsw/hnsw.hpp"
#include "utils/io.hpp"
#include "utils/stopw.hpp"

DEFINE_int32(num_threads, 1, "Number of threads to use");
DEFINE_string(output_csv_path, "", "Output CSV file path");

// std::vector<size_t> efs = {10,  20,  40,  50,  60,  80,  100, 150,  170,  190, 200,
//                            250, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000};
std::vector<size_t> efs = {400};

size_t test_round = 3;
size_t topk = 100;

using PID = rabitqlib::PID;
using index_type = rabitqlib::hnsw::HierarchicalNSW;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for index \n"
                  << "arg2: path for query file, format .fvecs\n"
                  << "arg3: path for groundtruth file format .ivecs\n"
                  << "arg4: metric type (\"l2\" or \"ip\")\n";
        exit(1);
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    char* index_file = argv[1];
    char* query_file = argv[2];
    char* gt_file = argv[3];
    // int FLAGS_num_threads = std::stoi(argv[5]);

    data_type query;
    gt_type gt;
    rabitqlib::load_something<float, data_type>(query_file, query);
    rabitqlib::load_something<uint32_t, gt_type>(gt_file, gt);
    size_t nq = query.rows();
    size_t total_count = nq * topk;

    index_type hnsw;
    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    if (argc > 4) {
        std::string metric_str(argv[4]);
        if (metric_str == "ip" || metric_str == "IP") {
            metric_type = rabitqlib::METRIC_IP;
        }
    }
    if (metric_type == rabitqlib::METRIC_IP) {
        std::cout << "Metric Type: IP\n";
    } else if (metric_type == rabitqlib::METRIC_L2) {
        std::cout << "Metric Type: L2\n";
    }

    hnsw.load(index_file, metric_type);

    auto nefs = efs;

    size_t length = nefs.size();
    std::cout << "search start >.....\n";

    auto run_q_f = [&](int num_threads){
        std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
        std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(length));
        std::vector<std::vector<float>> all_compute_kopps(test_round, std::vector<float>(length));
        std::vector<std::vector<float>> all_bandwisth_mbps(test_round, std::vector<float>(length));

        for (size_t i_probe = 0; i_probe < length; ++i_probe) {
            for (size_t r = 0; r < test_round; r++) {
                size_t ef = nefs[i_probe];
                size_t total_correct = 0;
                float total_time = 0;
                hnsw.query_runtime_metrics_.fast_bitsum = 0;
                hnsw.query_runtime_metrics_.acc_bitsum = 0;
                hnsw.query_runtime_metrics_.total_comp_cnt = 0;

                auto start = std::chrono::high_resolution_clock::now();

                std::vector<std::vector<std::pair<float, PID>>> res =
                    hnsw.search(query.data(), nq, topk, ef, num_threads);

                auto end = std::chrono::high_resolution_clock::now();

                float elapsed_us =
                    std::chrono::duration<float, std::micro>(end - start).count();

                total_time += elapsed_us;

                for (size_t i = 0; i < nq; i++) {
                    for (size_t j = 0; j < topk; j++) {
                        for (size_t k = 0; k < topk; k++) {
                            if (gt(i, k) == res[i][j].second) {
                                total_correct++;
                                break;
                            }
                        }
                    }
                }

                float qps = static_cast<float>(nq) / ((total_time) / 1e6F);

                float recall =
                    static_cast<float>(total_correct) / static_cast<float>(total_count);

                all_qps[r][i_probe] = qps;
                all_recall[r][i_probe] = recall;
                all_compute_kopps[r][i_probe] = hnsw.query_runtime_metrics_.total_comp_cnt / 1000.0 / ((total_time) / 1e6F);
                all_bandwisth_mbps[r][i_probe] = (hnsw.query_runtime_metrics_.fast_bitsum + hnsw.query_runtime_metrics_.acc_bitsum) / 8.0 / (1<<20) / ((total_time) / 1e6F);
            }
        }

        auto avg_qps = rabitqlib::horizontal_avg(all_qps);
        auto avg_recall = rabitqlib::horizontal_avg(all_recall);
        auto avg_compute_kopps = rabitqlib::horizontal_avg(all_compute_kopps);
        auto avg_bandwisth_mbps = rabitqlib::horizontal_avg(all_bandwisth_mbps);

        std::ostringstream oss;
        for (size_t i = 0; i < avg_qps.size(); ++i) {
            oss << efs[i] << "," << num_threads << "," << avg_qps[i] << "," << avg_recall[i] << "," << avg_compute_kopps[i] << "," << avg_bandwisth_mbps[i] << "," << '\n';
        }

        const std::string output_text = oss.str();
        std::cout << output_text;
        return output_text;
    };

    std::string output_text = "EF,num_threads,QPS,Recall,compute_kopps,bw_mbps,\n";
    std::cout << output_text;
    if (FLAGS_num_threads){
        output_text += run_q_f(FLAGS_num_threads);
    } else{
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

    std::ofstream ofs(FLAGS_output_csv_path);
    if (ofs.is_open()) {
        ofs << output_text;
        ofs.close();
        std::cout << "Output CSV file: " << FLAGS_output_csv_path << "\n";
    } else {
        std::cerr << "Failed to open output CSV file: " << FLAGS_output_csv_path << "\n";
    }
}