#include "NumCpp.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
// #include <boost/filesystem.hpp>
#include <unistd.h>
// #include <Eigen/Dense>
#include <string.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

// bool comp(const nc::NdArrayIterator<int, int*, long int> &a, const nc::NdArrayIterator<int, int*, long int> &b)
// {
//     return  < ;
// }

// typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenIntMatrix;
// typedef Eigen::Map<EigenIntMatrix> EigenIntMatrixMap;
// Eigen::setNbThreads(16);


// int binarySearch(nc::NdArray<int>& edges, int L ,int R, int point_index)
// {
//     int mid;
//     int res = R+1;
//     while(L <= R)
//     {
//         mid =  L + (R - L) / 2;
//         cout << L << " " << R << " " << res << " " << edges(mid, 0) << endl;
//         if(edges(mid, 0) > point_index)
//         {
//             R = mid - 1;
//             res = mid;
//         }
//         else
//         {
//             L = mid+1;
//         }
//     }
//     return res;
// }


int binarySearch(nc::NdArray<int>& edges, int k, int last)
{
	int start = 0;
	while (start < last)
	{
		int mid = (start + last) / 2;
		if (edges(mid, 0) >= k)
		{
			last = mid;
		}
		else
		{
			start = mid + 1;
		}
	}
	return start;
}

nc::NdArray<int> get_neighbors_backup(nc::NdArray<int>& edges, int point_index)
{
    auto neighbor_index = nc::argwhere(edges == point_index).astype<int>();
    // neighbor_index.astype<int>();
    // neighbor_0_index is the neighbors index in edges.reshape(n,1)
    auto neighbor_0_index = nc::where(neighbor_index%2 == 1, neighbor_index/2, -1);
    neighbor_0_index = neighbor_0_index[neighbor_0_index>=0];
    // auto tmp = neighbor_0_index.reshape(neighbor_0_index.shape().cols, 1);
    // edges
    nc::NdArray<int> neighbors = {-1};
    for (int i = 0; i < neighbor_0_index.numCols(); i++)
    {
        nc::NdArray<int> tmp = {edges(neighbor_0_index(0, i), 0)};
        neighbors = nc::append(neighbors, tmp, nc::Axis::COL);
    }
    neighbors = neighbors[neighbors>=0];
    // cout << neighbors << endl;

    return neighbors;
}

nc::NdArray<int> get_neighbors(nc::NdArray<int>& edges, int point_index)
{
    int edges_num = edges.shape().rows;
    int edges_start = max(6*point_index - 333, 0);
    int edges_end = min(6*point_index + 333, edges_num);
    
    auto edges_small = edges(nc::Slice(edges_start, edges_end), nc::Slice(0, 2));
    int edges_small_num = edges_small.shape().rows;
    int idx_start = binarySearch(edges_small, point_index, edges_small_num);
    int idx_end = min(idx_start+20, edges_small_num);

    auto edges_6 = edges_small(nc::Slice(idx_start, idx_end), nc::Slice(0, 2));
    auto neighbor_index = nc::argwhere(edges_6 == point_index).astype<int>();


    // auto neighbor_index = nc::argwhere(edges == point_index).astype<int>();
    // cout << neighbor_index << endl << neighbor_index2 << endl << endl;

    // neighbor_index.astype<int>();
    // neighbor_0_index is the neighbors index in edges.reshape(n,1)
    auto neighbor_0_index = nc::where(neighbor_index%2 == 0, neighbor_index/2, -1);
    neighbor_0_index = neighbor_0_index[neighbor_0_index>=0];
    // auto tmp = neighbor_0_index.reshape(neighbor_0_index.shape().cols, 1);
    // edges
    nc::NdArray<int> neighbors = {-1};
    for (int i = 0; i < neighbor_0_index.numCols(); i++)
    {
        nc::NdArray<int> tmp = {edges_6(neighbor_0_index(0, i), 1)};
        neighbors = nc::append(neighbors, tmp, nc::Axis::COL);
    }
    neighbors = neighbors[neighbors>=0];

    return neighbors;
}

void gemm_K(vector<int>& idx_i, vector<int>& idx_j, vector<int>& idx_v, int x_i, int x_j, int x_k, int type)
{
    if (type == 6)
    {
        idx_i.push_back(x_i);
        idx_j.push_back(x_i);
        idx_v.push_back(4);
        // K(x_i, x_i) += 4;
        idx_i.push_back(x_i);
        idx_j.push_back(x_j);
        idx_v.push_back(-8);
        // K(x_i, x_j) -= 8;
        idx_i.push_back(x_i);
        idx_j.push_back(x_k);
        idx_v.push_back(4);
        // K(x_i, x_k) += 4;
        idx_i.push_back(x_j);
        idx_j.push_back(x_i);
        idx_v.push_back(-8);
        // K(x_j, x_i) -= 8;
        idx_i.push_back(x_j);
        idx_j.push_back(x_j);
        idx_v.push_back(16);
        // K(x_j, x_j) += 16;
        idx_i.push_back(x_j);
        idx_j.push_back(x_k);
        idx_v.push_back(-8);
        // K(x_j, x_k) -= 8;
        idx_i.push_back(x_k);
        idx_j.push_back(x_i);
        idx_v.push_back(4);
        // K(x_k, x_i) += 4;
        idx_i.push_back(x_k);
        idx_j.push_back(x_j);
        idx_v.push_back(-8);
        // K(x_k, x_j) -= 8;
        idx_i.push_back(x_k);
        idx_j.push_back(x_k);
        idx_v.push_back(4);
        // K(x_k, x_k) += 4;
    }
    else if (type == 5)
    {
        idx_i.push_back(x_i);
        idx_j.push_back(x_i);
        idx_v.push_back(1);
        idx_i.push_back(x_i);
        idx_j.push_back(x_j);
        idx_v.push_back(-2);
        idx_i.push_back(x_i);
        idx_j.push_back(x_k);
        idx_v.push_back(1);
        idx_i.push_back(x_j);
        idx_j.push_back(x_i);
        idx_v.push_back(-2);
        idx_i.push_back(x_j);
        idx_j.push_back(x_j);
        idx_v.push_back(4);
        idx_i.push_back(x_j);
        idx_j.push_back(x_k);
        idx_v.push_back(-2);
        idx_i.push_back(x_k);
        idx_j.push_back(x_i);
        idx_v.push_back(1);
        idx_i.push_back(x_k);
        idx_j.push_back(x_j);
        idx_v.push_back(-2);
        idx_i.push_back(x_k);
        idx_j.push_back(x_k);
        idx_v.push_back(1);
    }
}

vector<vector<int>> compute_K(nc::NdArray<int>& edges, int vertices_num)
{
    // auto K = nc::zeros<nc::int8>(vertices_num, vertices_num);

    // auto S = nc::zeros<nc::int8>(vertices_num, vertices_num);
    // auto C_k = nc::zeros<nc::int8>(vertices_num, 1);
    // auto C_j = nc::zeros<nc::int8>(vertices_num, 1);


    // cv::Mat cv_K = cv::Mat(K.numRows(), K.numCols(), CV_8SC1, K.data());
    // auto eigenK = EigenIntMatrixMap(K.data(), K.numRows(), K.numCols());
    vector<int> idx_i, idx_j, idx_v;

    int v6 = 0;
    for (int vertice = 0; vertice < vertices_num; vertice++)
    {
        cout << vertice << endl;
        int x_i, x_j, x_k;

        nc::NdArray<int> vertice_neighbors = get_neighbors(edges, vertice);
        // for (int idx = 0; idx < vertice_neighbors.numCols(); idx++)
        // {
        //     x_k = vertice;
        //     x_j = vertice_neighbors(0, idx);
        //     auto cjk = (1 - C_j(x_j, 0)) * (1 - C_k(x_k, 0));
        //     S(x_k, x_k) += cjk;
        //     S(x_k, x_j) -= cjk;
        //     S(x_j, x_k) -= cjk;
        //     S(x_j, x_j) += cjk;
        // }
        
        // cout << "point: " << vertice << " neighbor: " << vertice_neighbors << endl;

        nc::NdArray<int> sorted_vertice_neighbors = nc::copy(vertice_neighbors);
        if (vertice_neighbors.numCols() == 6)
        {
            try
            {
                nc::NdArray<int> neighbor0_neighbors = get_neighbors(edges, sorted_vertice_neighbors(0, 0));
                nc::NdArray<int> tmp = nc::intersect1d(vertice_neighbors, neighbor0_neighbors);
                sorted_vertice_neighbors(0, 1) = tmp(0, 0);
                sorted_vertice_neighbors(0, 2) = tmp(0, 1);
                // cout << "point: " << sorted_vertice_neighbors(0, 0) << " neighbor: " << neighbor0_neighbors << " tmp: " << tmp << endl;

                nc::NdArray<int> neighbor1_neighbors = get_neighbors(edges, sorted_vertice_neighbors(0, 1));
                neighbor1_neighbors = nc::where(neighbor1_neighbors != sorted_vertice_neighbors(0, 0), neighbor1_neighbors, -1);
                neighbor1_neighbors = neighbor1_neighbors[neighbor1_neighbors>=0];
                tmp = nc::intersect1d(vertice_neighbors, neighbor1_neighbors);
                sorted_vertice_neighbors(0, 3) = tmp(0, 0);

                nc::NdArray<int> neighbor2_neighbors = get_neighbors(edges, sorted_vertice_neighbors(0, 2));
                neighbor2_neighbors = nc::where(neighbor2_neighbors != sorted_vertice_neighbors(0, 0), neighbor2_neighbors, -1);
                neighbor2_neighbors = neighbor2_neighbors[neighbor2_neighbors>=0];
                tmp = nc::intersect1d(vertice_neighbors, neighbor2_neighbors);
                sorted_vertice_neighbors(0, 4) = tmp(0, 0);

                nc::NdArray<int> neighbor3_neighbors = get_neighbors(edges, sorted_vertice_neighbors(0, 3));
                neighbor3_neighbors = nc::where(neighbor3_neighbors != sorted_vertice_neighbors(0, 1), neighbor3_neighbors, -1);
                neighbor3_neighbors = neighbor3_neighbors[neighbor3_neighbors>=0];
                tmp = nc::intersect1d(vertice_neighbors, neighbor3_neighbors);
                sorted_vertice_neighbors(0, 5) = tmp(0, 0);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                continue;
            }
            // nc::dot could be replaced as cv:Mat or eigen
            for (int idx = 0; idx < 3; idx++)
            {
                x_i = sorted_vertice_neighbors(0, 2*idx);
                x_k = sorted_vertice_neighbors(0, 5-2*idx);
                x_j = vertice;
                gemm_K(idx_i, idx_j, idx_v, x_i, x_j, x_k, 6);

                // cout << K.astype<int>() << endl;
                // auto A = nc::zeros<nc::int8>(vertices_num, 1);
                // A(sorted_vertice_neighbors(0, 2*idx), 0) = A(sorted_vertice_neighbors(0, 5-2*idx), 0) = 2;
                // A(vertice, 0) = -4;

                // auto eigenA = EigenIntMatrixMap(A.data(), A.numRows(), A.numCols());
                // auto eigenAT = EigenIntMatrixMap(A.data(), A.numCols(), A.numRows());
                // auto eigentmpK = eigenA * eigenAT;


                // cv::Mat cv_A = cv::Mat(A.numRows(), A.numCols(), CV_32FC1, A.data());
                // cv::Mat cv_AT = cv::Mat(A.numCols(), A.numRows(), CV_32FC1, A.data());
                // // cv::Mat cv_tmpK = cv_A * cv_AT;
                // cv::Mat cv_tmpK;
                // cv::cuda::gemm(cv_A, cv_AT, 1, cv::Mat(), 1, cv_K);

                // cv::transpose(cv_A, cv_AT);
                
                // cv_tmpK.convertTo(cv_tmpK, CV_8SC1);
                // auto tmpK = nc::NdArray<nc::uint8>(cv_tmpK.data, cv_tmpK.rows, cv_tmpK.cols).astype<nc::int8>();

                // cv_K = cv_K + cv_tmpK;
                // cv::cuda::add(cv_K, cv_tmpK, cv_K);
                // K = K + tmpK;
            }
        }
        else if (vertice_neighbors.numCols() == 5)
        {
            try
            {
                nc::NdArray<int> neighbor0_neighbors = get_neighbors(edges, sorted_vertice_neighbors(0, 0));
                nc::NdArray<int> tmp = nc::intersect1d(vertice_neighbors, neighbor0_neighbors);
                sorted_vertice_neighbors(0, 1) = tmp(0, 0);
                sorted_vertice_neighbors(0, 2) = tmp(0, 1);

                nc::NdArray<int> neighbor1_neighbors = get_neighbors(edges, sorted_vertice_neighbors(0, 1));
                neighbor1_neighbors = nc::where(neighbor1_neighbors != sorted_vertice_neighbors(0, 0), neighbor1_neighbors, -1);
                neighbor1_neighbors = neighbor1_neighbors[neighbor1_neighbors>=0];
                tmp = nc::intersect1d(vertice_neighbors, neighbor1_neighbors);
                sorted_vertice_neighbors(0, 3) = tmp(0, 0);

                nc::NdArray<int> neighbor2_neighbors = get_neighbors(edges, sorted_vertice_neighbors(0, 2));
                neighbor2_neighbors = nc::where(neighbor2_neighbors != sorted_vertice_neighbors(0, 0), neighbor2_neighbors, -1);
                neighbor2_neighbors = neighbor2_neighbors[neighbor2_neighbors>=0];
                tmp = nc::intersect1d(vertice_neighbors, neighbor2_neighbors);
                sorted_vertice_neighbors(0, 4) = tmp(0, 0);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                continue;
            }
            x_j = vertice;
            x_i = sorted_vertice_neighbors(0, 0);
            x_k = sorted_vertice_neighbors(0, 3);
            gemm_K(idx_i, idx_j, idx_v, x_i, x_j, x_k, 5);

            x_i = sorted_vertice_neighbors(0, 0);
            x_k = sorted_vertice_neighbors(0, 4);
            gemm_K(idx_i, idx_j, idx_v, x_i, x_j, x_k, 5);
            
            x_i = sorted_vertice_neighbors(0, 1);
            x_k = sorted_vertice_neighbors(0, 2);
            gemm_K(idx_i, idx_j, idx_v, x_i, x_j, x_k, 5);

            x_i = sorted_vertice_neighbors(0, 1);
            x_k = sorted_vertice_neighbors(0, 4);
            gemm_K(idx_i, idx_j, idx_v, x_i, x_j, x_k, 5);

            x_i = sorted_vertice_neighbors(0, 2);
            x_k = sorted_vertice_neighbors(0, 3);
            gemm_K(idx_i, idx_j, idx_v, x_i, x_j, x_k, 5);

        }
    }

    // cout << K.astype<int>() << endl;
    // cout << S.astype<int>() << endl;
    
    // string saveTxt = "/workspace/nvrender/results/scan24/K.txt";
    // cout << saveTxt << endl;
    // const char separator = 32;
    // nc::tofile(K.astype<int>(), saveTxt, separator);


    vector<vector<int>> result;
    // for (int vertice = 0; vertice < vertices_num; vertice++)
    // {
    //     nc::NdArray<int> vertice_neighbors = get_neighbors(edges, vertice);
    //     if (K(vertice, vertice) != 0){
    //         idx_i.push_back(vertice);
    //         idx_j.push_back(vertice);
    //         idx_v.push_back(K(vertice, vertice));
    //         K(vertice, vertice) = 0;
    //     }
    //     for (int ni = 0; ni < vertice_neighbors.numCols(); ni++)
    //     {
    //         if (K(vertice, vertice_neighbors(0, ni)) != 0){
    //             idx_i.push_back(vertice);
    //             idx_j.push_back(vertice_neighbors(0, ni));
    //             idx_v.push_back(K(vertice, vertice_neighbors(0, ni)));
    //             K(vertice_neighbors(0, ni), vertice) = 0;
    //         }
    //         for (int vni = 0; vni < vertice_neighbors.numCols(); vni++)
    //         {
    //             int i = vertice_neighbors(0, ni);
    //             int j = vertice_neighbors(0, vni);
    //             if (K(i,j) != 0)
    //             {
    //                 idx_i.push_back(i);
    //                 idx_j.push_back(j);
    //                 idx_v.push_back(K(i, j));
    //                 K(i, j) = 0;
    //             }
    //         }
    //     }
    // }

    result.push_back(idx_i);
    result.push_back(idx_j);
    result.push_back(idx_v);

    return result;
    // cout << v6 << endl;
}

int main()
{
    const char separator = 32;
    // auto tempDir = boost::filesystem::current_path();
    string edgesDir = getcwd(NULL, 0);
    // string edges2Txt = (edgesDir + "/edges.txt");
    string edges2Txt = "/workspace/nvrender/results/scan24/edges.txt";
    cout << edges2Txt << endl;
    auto edges = nc::fromfile<int>(edges2Txt, separator).reshape(-1, 2);
    string vertices_num2Txt = "/workspace/nvrender/results/scan24/vertices.txt";
    auto vertices_num_txt = nc::fromfile<int>(vertices_num2Txt, separator).reshape(-1, 2);
    int vertices_num = vertices_num_txt(0, 0);
    cout << vertices_num << endl;

    // nc::NdArray<int> edges = {{ 0,  2}, { 0,  4}, { 0,  6}, { 0,  8}, { 1,  3}, { 1,  5}, { 1,  7}, { 1,  8}, { 1,  9}, { 1, 11}, { 2,  0}, { 2,  6}, { 2,  8}, { 2,  9}, { 3,  1}, { 3,  7}, { 3,  9}, { 3, 10}, { 4,  0}, { 4,  6}, { 4,  8}, { 4, 11}, { 5,  1}, { 5,  7}, { 5, 10}, { 5, 11}, { 6,  0}, { 6,  2}, { 6,  4}, { 6,  9}, { 6, 10}, { 6, 11}, { 7,  1}, { 7,  3}, { 7,  5}, { 7, 10}, { 8,  0}, { 8,  1}, { 8,  2}, { 8,  4}, { 8,  9}, { 8, 11}, { 9,  1}, { 9,  2}, { 9,  3}, { 9,  6}, { 9,  8}, { 9, 10}, {10,  3}, {10,  5}, {10,  6}, {10,  7}, {10,  9}, {10, 11}, {11,  1}, {11,  4}, {11,  5}, {11,  6}, {11,  8}, {11, 10}};
    // // cout << edges.shape() << endl;
    // int vertices_num = 12;


    clock_t start,end;
    start = clock();
    vector<vector<int>> result;
    result = compute_K(edges, vertices_num);    
    end = clock();
    cout<<"time cost "<< double(end-start)/CLOCKS_PER_SEC << "s" <<endl;
}

vector<vector<int>> compute_k(const string& scan){
    const char separator = 32;
    string edgesDir = getcwd(NULL, 0);
    // string edges2Txt = (edgesDir + "/edges.txt");
    string edges2Txt = "/workspace/nvrender/results/" + scan + "/edges.txt";
    auto edges = nc::fromfile<int>(edges2Txt, separator).reshape(-1, 2);
    string vertices_num2Txt = "/workspace/nvrender/results/" + scan + "/vertices.txt";
    auto vertices_num_txt = nc::fromfile<int>(vertices_num2Txt, separator).reshape(-1, 2);
    int vertices_num = vertices_num_txt(0, 0);
    cout << vertices_num << endl;
    vector<vector<int>> result;
    result = compute_K(edges, vertices_num);    
    return result;
}

PYBIND11_MODULE( py2cpp, m ){
    m.doc() = "compute K sparse matrix";
    m.def("compute_k", &compute_k, "return vector" );
    // g++ -shared -fPIC `python -m pybind11 --includes` computeK.cpp -o py2cpp.so -I /opt/conda/include/python3.8/
}