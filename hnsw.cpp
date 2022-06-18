#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include <unordered_set>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace hnswlib;

float comput_recall(vector<vector<unsigned>>& result,
                    vector<vector<unsigned>>& massQA, size_t qsize, size_t k){
    size_t correct = 0;
    size_t total = qsize * k;

    for (int qi = 0; qi < qsize; qi++){
        unordered_set<unsigned> g;
        for (int i = 0; i < k; i++)
            g.insert(result[qi][i]);

        for (int i = 0; i < k; i++) {
            if (g.find(massQA[qi][i]) != g.end())
                correct++;
        }
    }
    return (1.0 * correct / total);
}

template<typename DTset, typename DTres>
static void
test_vs_recall(HierarchicalNSW<DTset, DTres>& appr_alg, size_t vecdim,
                DTset *massQ, size_t qsize,
                vector<vector<unsigned>>& massQA, size_t k,
                map<size_t, vector<float>>& mapResult) {
    vector<size_t> efs;// = { 10,10,10,10,10 };
    for (int i = 10; i <= 150; i += 10)
        efs.push_back(i);

    int column_map = 2;

    cout << "efs\t" << "R@" << k << "\t" << "time_us\t";
#if RANKMAP
    if (appr_alg.stats != nullptr) {
        cout << "rank_us\t" << "sort_us\t" << "hlc_us\t" << "visited_us\t";
        cout << "NDC_max\t" << "NDC_total\t";
        column_map = 8;
    }
#else
    cout << "n_hop_L\t" << "n_hop_0\t" << "NDC\t";
#endif
    cout << endl;

    vector<float> result_ef(column_map);

    for (size_t ef : efs) {
        appr_alg.setEf(ef);

#if RANKMAP
        if (appr_alg.stats != nullptr)
            appr_alg.stats->Reset();
#else
        appr_alg.metric_hops = 0;
        appr_alg.metric_hops_L = 0;
        appr_alg.metric_distance_computations = 0;
#endif

        vector<vector<unsigned>> result(qsize);
        for (vector<unsigned>& r: result)
            r.resize(k, 0);

        Timer stopw = Timer();
//         omp_set_num_threads(3);
// #pragma omp parallel for
        for (int qi = 0; qi < qsize; qi++) {
#if RANKMAP
            priority_queue<pair<DTres, labeltype>> res = appr_alg.searchParaRank(massQ + vecdim * qi, k);
#else
            priority_queue<pair<DTres, labeltype>> res = appr_alg.searchKnn(massQ + vecdim * qi, k);
#endif

// #pragma omp critical
            {
                int i = 0;
                while (!res.empty()){
                    result[qi][i] = (unsigned) res.top().second;
                    res.pop();
                    i++;
                }
            }
        }
        float time_us_per_query = stopw.getElapsedTimeus() / qsize;
        float recall = comput_recall(result, massQA, qsize, k);

        cout << ef << "\t" << recall << "\t" << time_us_per_query << "\t";
        result_ef[0] = recall;
        result_ef[1] = time_us_per_query;
#if RANKMAP
        if (appr_alg.stats != nullptr) {
            cout << appr_alg.stats->rank_us / qsize << "\t";
            cout << appr_alg.stats->sort_us / qsize << "\t";
            cout << appr_alg.stats->hlc_us / qsize << "\t";
            cout << appr_alg.stats->visited_us / qsize << "\t";
            result_ef[2] = appr_alg.stats->rank_us / qsize;
            result_ef[3] = appr_alg.stats->sort_us / qsize;
            result_ef[4] = appr_alg.stats->hlc_us / qsize;
            result_ef[5] = appr_alg.stats->visited_us / qsize;

            cout << (1.0 * appr_alg.stats->n_DC_max / qsize) << "\t";
            cout << (1.0 * appr_alg.stats->n_DC_total / qsize) << "\t";
            result_ef[6] = 1.0 * appr_alg.stats->n_DC_max / qsize;
            result_ef[7] = 1.0 * appr_alg.stats->n_DC_total / qsize;
        }
#else
        cout << (1.0 * appr_alg.metric_hops_L / qsize) << "\t";
        cout << (1.0 * appr_alg.metric_hops / qsize) << "\t";
        cout << (1.0 * appr_alg.metric_distance_computations / qsize) << "\t";
#endif
        cout << endl;

        mapResult.insert(pair<size_t, vector<float>>(ef, result_ef));

        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const string &name) {
    ifstream f(name.c_str());
    return f.good();
}

template<typename DTset, typename DTres>
void build_index(map<string, size_t> &MapParameter, map<string, string> &MapString, bool isSave = true){
    //
    size_t efConstruction = MapParameter["efConstruction"];
    size_t M = MapParameter["M"];
    size_t vecsize = MapParameter["vecsize"];
    size_t vecdim = MapParameter["vecdim"];
    size_t qsize = MapParameter["qsize"];

    string path_data = MapString["path_data"];
    string index = MapString["index"];

    if (exists_test(index)){
        printf("Index %s is existed \n", index.c_str());
        return;
    } else {

        DTset *massB = new DTset[vecsize * vecdim]();
        cout << "Loading base data:\n";
        LoadBinToArray<DTset>(path_data, massB, vecsize, vecdim);

#if FMTINT
        L2SpaceI l2space(vecdim);
#else
        L2Space l2space(vecdim);
#endif
        HierarchicalNSW<DTset, DTres> *appr_alg = new HierarchicalNSW<DTset, DTres>(&l2space, vecsize, M, efConstruction);
#if PLATG
        unsigned center_id = compArrayCenter<DTset>(massB, vecsize, vecdim);
        appr_alg->addPoint((void *) (massB + center_id * vecdim), (size_t) center_id);
#else
        appr_alg->addPoint((void *) (massB), (size_t) 0);
#endif
        cout << "Building index:\n";
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = vecsize / 10;
#pragma omp parallel for
        for (size_t i = 1; i < vecsize; i++) {
#pragma omp critical
            {
                j1++;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * stopw.getElapsedTimes()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
#if PLATG
            size_t ic;
            if (i <= center_id)
                ic = i - 1;
            else
                ic = i;
            appr_alg->addPoint((void *) (massB + ic * vecdim), ic);
#else
            appr_alg->addPoint((void *) (massB + i * vecdim), i);
#endif
        }
        cout << "Build time:" << stopw_full.getElapsedTimes() << "  seconds\n";
        delete[] massB;
        if (isSave)
            appr_alg->saveIndex(index);

        printf("Build index %s is succeed \n", index.c_str());
    }
}

template<typename DTset, typename DTres>
void search_index(map<string, size_t> &MapParameter, map<string, string> &MapString){
    //
    size_t k = MapParameter["k"];
    size_t vecsize = MapParameter["vecsize"];
    size_t qsize = MapParameter["qsize"];
    size_t vecdim = MapParameter["vecdim"];
    size_t gt_maxnum = MapParameter["gt_maxnum"];

    string path_q = MapString["path_q"];
    string index = MapString["index"];
    string path_gt = MapString["path_gt"];

    if (!exists_test(index)){
        printf("Error, index %s is unexisted \n", index.c_str());
        exit(1);
    } else {

        vector<vector<unsigned>> massQA;
        DTset *massQ = new DTset[qsize * vecdim];

        cout << "Loading GT:\n";
        LoadBinToVector<unsigned>(path_gt, massQA, qsize, gt_maxnum);
        cout << "Loading queries:\n";
        LoadBinToArray<DTset>(path_q, massQ, qsize, vecdim);

#if FMTINT
        L2SpaceI l2space(vecdim);
#else
        L2Space l2space(vecdim);
#endif
        HierarchicalNSW<DTset, DTres> *appr_alg = new HierarchicalNSW<DTset, DTres>(&l2space, index, false);

#if RANKMAP
        appr_alg->initRankMap();
#endif

        int test_times = TTIMES;
        printf("[Test Mode] Run %d times and comput recall: \n", test_times);
        vector<map<size_t, vector<float>>> finishedResult(test_times);
        for (int i = 0; i < test_times; i++) {
            test_vs_recall(*appr_alg, vecdim, massQ, qsize, massQA, k, finishedResult[i]);
        }

        ofstream result_writer(MapString["result"].c_str());
        result_writer << "efs\t" << "R@" << k << "\t" << "time_us\t";
#if RANKMAP
        if (appr_alg->stats != nullptr) {
            result_writer << "rank_us\t" << "sort_us\t" << "hlc_us\t" << "visited_us\t";
            result_writer << "NDC_max\t" << "NDC_total\t";
        }
#endif
        result_writer << "\n";

        for (auto iter = finishedResult[0].begin(); iter != finishedResult[0].end(); iter++) {
            size_t ef = iter->first;
            vector<float> recall_list(test_times);
            for (int ri = 0; ri < test_times; ri++){
                recall_list[ri] = finishedResult[ri][ef][1];
            }
            int pos = selectNearAvgPos(recall_list);

            vector<float> res_selected(finishedResult[pos][ef]);
            result_writer << ef << "\t";
            for (float& r : res_selected)
                result_writer << r << "\t";
            result_writer << "\n";
        }
        result_writer.close();

        printf("Search index %s is succeed \n", index.c_str());
    }
}

void hnsw_impl(string stage, string using_dataset, size_t data_size_millions){
    string path_project = "..";
#if RANKMAP
    string label = "rank-map/";
#else
    string label = "plat/";
#endif

    string path_graphindex = path_project + "/graphindex/" + label;

    string pre_index = path_graphindex + using_dataset;
    if (access(pre_index.c_str(), R_OK|W_OK)){
        if (mkdir(pre_index.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", pre_index.c_str());
            exit(1);
        }
    }

    // for 1m, 10m, 100m
    vector<size_t> efcSet = {20, 30, 40};
    size_t M = (log10(data_size_millions) + 2) * 10;
	size_t efConstruction = M * 10;
    size_t k = 10;
    size_t vecsize = data_size_millions * 1000000;

    map<string, size_t> MapParameter;
    MapParameter["data_size_millions"] = data_size_millions;
    MapParameter["efConstruction"] = efConstruction;
    MapParameter["M"] = M;
    MapParameter["k"] = k;
    MapParameter["vecsize"] = vecsize;

    map<string, string> MapString;

    string hnsw_index = pre_index + "/" + using_dataset + to_string(data_size_millions) +
                        "m_ef" + to_string(efConstruction) + "m" + to_string(M) + ".bin";
    MapString["index"] = hnsw_index;
    CheckDataset(using_dataset, MapParameter, MapString);

#if RANKMAP
    string suffix = "";
#if TESTMODE
    suffix = "detail";
#endif
    MapString["result"] = path_project + "/output/result/rank/" + using_dataset + to_string(data_size_millions) +
                        "m_rc" + to_string(k) + "_rank" + to_string(NUM_RANKS) + "_" + suffix + ".log";
#else
    MapString["result"] = path_project + "/output/result/hnsw/" + using_dataset + to_string(data_size_millions) +
                        "m_rc" + to_string(k) + ".log";
#endif

    if (stage == "build" || stage == "both")
        build_index<DTSET, DTRES>(MapParameter, MapString);

    if (stage == "search" || stage == "both")
        search_index<DTSET, DTRES>(MapParameter, MapString);

    return;
}
