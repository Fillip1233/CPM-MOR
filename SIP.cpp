#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

struct SparseMatrices {
    SparseMatrix<double> G_hat;
    SparseMatrix<double> C_hat;
    SparseMatrix<double> E_hat;
};

SparseMatrices SIPcore3(const SparseMatrix<double>& G, 
                       const SparseMatrix<double>& C, 
                       const SparseMatrix<double>& E,
                       const vector<int>& ports, 
                       double threshold = 2.0) {
    
    int n = G.rows();
    int m = ports.size();
    
    auto start_time = high_resolution_clock::now();
    
    // Create permutation vector
    vector<bool> is_port(n, false);
    for (int port : ports) {
        if (port < n) {
            is_port[port] = true;
        }
    }
    
    vector<int> non_ports;
    for (int i = 0; i < n; ++i) {
        if (!is_port[i]) {
            non_ports.push_back(i);
        }
    }
    
    vector<int> perm;
    perm.reserve(n);
    perm.insert(perm.end(), non_ports.begin(), non_ports.end());
    perm.insert(perm.end(), ports.begin(), ports.end());
    
    // Permute matrices
    vector<Triplet<double>> triplets_G, triplets_C, triplets_E;
    triplets_G.reserve(G.nonZeros());
    triplets_C.reserve(C.nonZeros());
    triplets_E.reserve(E.nonZeros());
    
    for (int i = 0; i < n; ++i) {
        int new_i = find(perm.begin(), perm.end(), i) - perm.begin();
        
        // Permute G
        for (SparseMatrix<double>::InnerIterator it(G, i); it; ++it) {
            int new_j = find(perm.begin(), perm.end(), it.col()) - perm.begin();
            triplets_G.emplace_back(new_i, new_j, it.value());
        }
        
        // Permute C
        for (SparseMatrix<double>::InnerIterator it(C, i); it; ++it) {
            int new_j = find(perm.begin(), perm.end(), it.col()) - perm.begin();
            triplets_C.emplace_back(new_i, new_j, it.value());
        }
        
        // Permute E
        for (SparseMatrix<double>::InnerIterator it(E, i); it; ++it) {
            triplets_E.emplace_back(new_i, it.col(), it.value());
        }
    }
    
    SparseMatrix<double> G_perm(n, n);
    SparseMatrix<double> C_perm(n, n);
    SparseMatrix<double> E_perm(n, E.cols());
    
    G_perm.setFromTriplets(triplets_G.begin(), triplets_G.end());
    C_perm.setFromTriplets(triplets_C.begin(), triplets_C.end());
    E_perm.setFromTriplets(triplets_E.begin(), triplets_E.end());
    
    // Add small diagonal to avoid singularity
    for (int i = 0; i < n; ++i) {
        C_perm.coeffRef(i, i) += 1e-15;
    }
    
    auto perm_time = high_resolution_clock::now();
    cout << "Permutation time: " 
         << duration_cast<milliseconds>(perm_time - start_time).count() / 1000.0 
         << " seconds" << endl;
    
    // Initialize submatrices
    SparseMatrix<double> subG = G_perm;
    SparseMatrix<double> subC = C_perm;
    
    for (int i = 0; i < n - m; ++i) {
        int k = n - i;
        
        auto col_start = high_resolution_clock::now();
        
        // Find pivot
        double max_val = 0.0;
        int pivot_idx = 0;
        
        for (int j = 0; j < k - m; ++j) {
            double diag_val = abs(subG.coeff(j, j));
            if (diag_val > max_val) {
                max_val = diag_val;
                pivot_idx = j;
            }
        }
        
        cout << "Processing column " << i + 1 << "/" << n - m 
             << "..., pivot value: " << max_val 
             << ", pivot index: " << pivot_idx << endl;
        
        if (max_val < threshold) {
            cout << "Column " << i + 1 
                 << ", pivot value is too small: " << max_val << endl;
            cout << "End of reduction, pivot value is too small." << endl;
            break;
        }
        
        // Swap pivot to position 0 if needed
        if (pivot_idx != 0) {
            vector<int> swap_indices(k);
            for (int j = 0; j < k; ++j) {
                swap_indices[j] = j;
            }
            swap(swap_indices[0], swap_indices[pivot_idx]);
            
            // Create permutation matrix for swapping
            SparseMatrix<double> P(k, k);
            vector<Triplet<double>> P_triplets;
            P_triplets.reserve(k);
            for (int j = 0; j < k; ++j) {
                P_triplets.emplace_back(swap_indices[j], j, 1.0);
            }
            P.setFromTriplets(P_triplets.begin(), P_triplets.end());
            
            subG = P.transpose() * subG * P;
            subC = P.transpose() * subC * P;
        }
        
        double a_ii = subG.coeff(0, 0);
        
        // Extract row 0, columns 1 to k-1
        VectorXd b_i(k - 1);
        for (int j = 1; j < k; ++j) {
            b_i(j - 1) = subG.coeff(0, j);
        }
        
        // Create M matrix
        SparseMatrix<double> M(k, k);
        vector<Triplet<double>> M_triplets;
        M_triplets.reserve(k + k - 1);
        
        // Diagonal entries
        for (int j = 0; j < k; ++j) {
            M_triplets.emplace_back(j, j, 1.0);
        }
        
        // Off-diagonal entries
        for (int j = 1; j < k; ++j) {
            M_triplets.emplace_back(0, j, -b_i(j - 1) / a_ii);
        }
        
        M.setFromTriplets(M_triplets.begin(), M_triplets.end());
        
        // Apply transformation
        subG = M.transpose() * subG * M;
        subC = M.transpose() * subC * M;
        
        // Extract submatrices (remove first row and column)
        SparseMatrix<double> new_subG(k - 1, k - 1);
        SparseMatrix<double> new_subC(k - 1, k - 1);
        
        vector<Triplet<double>> new_G_triplets, new_C_triplets;
        new_G_triplets.reserve(subG.nonZeros());
        new_C_triplets.reserve(subC.nonZeros());
        
        for (int j = 1; j < k; ++j) {
            for (SparseMatrix<double>::InnerIterator it(subG, j); it; ++it) {
                if (it.col() >= 1) {
                    new_G_triplets.emplace_back(j - 1, it.col() - 1, it.value());
                }
            }
            
            for (SparseMatrix<double>::InnerIterator it(subC, j); it; ++it) {
                if (it.col() >= 1) {
                    new_C_triplets.emplace_back(j - 1, it.col() - 1, it.value());
                }
            }
        }
        
        new_subG.setFromTriplets(new_G_triplets.begin(), new_G_triplets.end());
        new_subC.setFromTriplets(new_C_triplets.begin(), new_C_triplets.end());
        
        subG = new_subG;
        subC = new_subC;
        
        auto col_end = high_resolution_clock::now();
        cout << "Time taken for column " << i + 1 << ": " 
             << duration_cast<milliseconds>(col_end - col_start).count() / 1000.0 
             << " seconds" << endl;
    }
    
    // Extract E_hat
    int reduced_size = subG.rows();
    SparseMatrix<double> E_hat(reduced_size, E.cols());
    vector<Triplet<double>> E_triplets;
    E_triplets.reserve(E_perm.nonZeros());
    
    for (int i = n - reduced_size; i < n; ++i) {
        for (SparseMatrix<double>::InnerIterator it(E_perm, i); it; ++it) {
            E_triplets.emplace_back(i - (n - reduced_size), it.col(), it.value());
        }
    }
    E_hat.setFromTriplets(E_triplets.begin(), E_triplets.end());
    
    auto total_time = high_resolution_clock::now();
    cout << "Total reduction time: " 
         << duration_cast<milliseconds>(total_time - start_time).count() / 1000.0 
         << " seconds" << endl;
    
    return {subG, subC, E_hat};
}