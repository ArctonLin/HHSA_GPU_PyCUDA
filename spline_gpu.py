import cupy as cp

# Kernels for finding extrema, building tridiagonal system, and solving it.
# We will use cupy.RawKernel for high performance batched operations.

extrema_kernel = cp.RawKernel(r'''
extern "C" __global__
void find_extrema(const float* data, int length, int num_ensembles,
                  bool* up_extrema, bool* low_extrema) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int ens = blockIdx.y;
    
    if (tid >= length || ens >= num_ensembles) return;
    
    int idx = ens * length + tid;
    
    // Clear extrema
    up_extrema[idx] = false;
    low_extrema[idx] = false;
    
    if (tid > 0 && tid < length - 1) {
        float prev = data[idx - 1];
        float curr = data[idx];
        float next = data[idx + 1];
        
        if (curr > prev && curr > next) {
            up_extrema[idx] = true;
        } else if (curr < prev && curr < next) {
            low_extrema[idx] = true;
        }
    }
}
''', 'find_extrema')

# To calculate Spline coefficients we need to build the tridiagonal matrix and RHS, 
# then solve it. Since number of extrema varies per ensemble, the matrix size varies!
# This makes purely batched Thomas algorithm tricky because lengths are different.
# We can instead implement a kernel that processes one ensemble per thread,
# extracting the X and Y arrays internally, building the system in local/global memory, and solving it.
# Because the number of extrema is bounded by length, we can allocate temporary global memory for the tridiagonal arrays.

spline_kernel = cp.RawKernel(r'''
extern "C" __global__
void batched_spline(const float* data, const bool* extrema, int length, int num_ensembles,
                    float* result, int type,
                    float* tmp_x, float* tmp_y,
                    float* tmp_a, float* tmp_b, float* tmp_c, float* tmp_rhs) {
                    
    int ens = blockIdx.x * blockDim.x + threadIdx.x;
    if (ens >= num_ensembles) return;
    
    // Pointers for this ensemble's temp memory
    float* X = tmp_x + ens * length;
    float* Y = tmp_y + ens * length;
    float* A_lower = tmp_a + ens * length; // i-1
    float* A_main = tmp_b + ens * length;  // i
    float* A_upper = tmp_c + ens * length; // i+1
    float* rhs = tmp_rhs + ens * length;
    
    // Extract X and Y
    int n = 0;
    if (type == 1) { // EMD
        X[n] = 0.0f;
        Y[n] = 0.0f;
        n++;
    }
    
    for (int i = 1; i < length - 1; ++i) {
        if (extrema[ens * length + i]) {
            X[n] = (float)i;
            Y[n] = data[ens * length + i];
            n++;
        }
    }
    
    if (type == 1) { // EMD
        X[n] = (float)(length - 1);
        Y[n] = 0.0f;
        n++;
    }
    
    // If not enough points, just copy data or set 0
    if (n < 3) {
        for (int i = 0; i < length; ++i) {
            result[ens * length + i] = (n > 0) ? Y[0] : 0.0f;
        }
        return;
    }
    
    // Build tridiagonal system for b
    // A(i, i-1) = A_lower, A(i, i) = A_main, A(i, i+1) = A_upper
    for (int i = 1; i < n - 1; ++i) {
        float dx1 = X[i] - X[i-1];
        float dx2 = X[i+1] - X[i];
        A_lower[i] = dx1 / 3.0f;
        A_main[i] = 2.0f * (dx1 + dx2) / 3.0f;
        A_upper[i] = dx2 / 3.0f;
        rhs[i] = (Y[i+1] - Y[i])/dx2 - (Y[i] - Y[i-1])/dx1;
    }
    
    // Boundary conditions (second derivative = 0)
    A_main[0] = 2.0f;
    A_upper[0] = 0.0f;
    rhs[0] = 0.0f;
    
    A_main[n-1] = 2.0f;
    A_lower[n-1] = 0.0f;
    rhs[n-1] = 0.0f;
    
    // Solve Tridiagonal System using Thomas Algorithm
    // Forward elimination
    for (int i = 1; i < n; ++i) {
        float m = A_lower[i] / A_main[i-1];
        A_main[i] -= m * A_upper[i-1];
        rhs[i] -= m * rhs[i-1];
    }
    
    // Backward substitution (we will store b in rhs)
    rhs[n-1] = rhs[n-1] / A_main[n-1];
    for (int i = n - 2; i >= 0; --i) {
        rhs[i] = (rhs[i] - A_upper[i] * rhs[i+1]) / A_main[i];
    }
    float* b = rhs; // alias
    
    // Calculate a and c in place, we can use A_lower for a, and A_upper for c
    float* a = A_lower;
    float* c = A_upper;
    for (int i = 0; i < n - 1; ++i) {
        float dx = X[i+1] - X[i];
        a[i] = (b[i+1] - b[i]) / (3.0f * dx);
        c[i] = (Y[i+1] - Y[i]) / dx - (2.0f * b[i] + b[i+1]) * dx / 3.0f;
    }
    
    // Spline evaluation for all points in [0, length-1]
    int idx = 0;
    for (int x = 0; x < length; ++x) {
        float px = (float)x;
        // find segment idx
        while (idx < n - 2 && px >= X[idx+1]) {
            idx++;
        }
        
        float h = px - X[idx];
        float val = 0.0f;
        
        if (px < X[0]) {
            // Left extrapolation
            float c0 = c[0];
            float b0 = b[0];
            val = (b0 * h + c0) * h + Y[0];
        } else if (px > X[n-1]) {
            // Right extrapolation
            float cn = c[n-2] + 2.0f * b[n-2] * (X[n-1]-X[n-2]) + 3.0f * a[n-2] * (X[n-1]-X[n-2]) * (X[n-1]-X[n-2]);
            val = (0.0f * h + cn) * h + Y[n-1];
        } else {
            val = ((a[idx] * h + b[idx]) * h + c[idx]) * h + Y[idx];
        }
        
        result[ens * length + x] = val;
    }
}
''', 'batched_spline')

class SplineGPU:
    def __init__(self, max_length, num_ensembles):
        self.max_length = max_length
        self.num_ensembles = num_ensembles
        
        # Allocate persistent temp buffers for Thomas algorithm to avoid reallocation
        self.tmp_x = cp.empty((num_ensembles, max_length), dtype=cp.float32)
        self.tmp_y = cp.empty((num_ensembles, max_length), dtype=cp.float32)
        self.tmp_a = cp.empty((num_ensembles, max_length), dtype=cp.float32)
        self.tmp_b = cp.empty((num_ensembles, max_length), dtype=cp.float32)
        self.tmp_c = cp.empty((num_ensembles, max_length), dtype=cp.float32)
        self.tmp_rhs = cp.empty((num_ensembles, max_length), dtype=cp.float32)

    def extract_extrema(self, data):
        num_ensembles, length = data.shape
        up_extrema = cp.zeros_like(data, dtype=cp.bool_)
        low_extrema = cp.zeros_like(data, dtype=cp.bool_)
        
        block_x = 256
        grid_x = (length + block_x - 1) // block_x
        grid_y = num_ensembles
        
        extrema_kernel((grid_x, grid_y), (block_x, 1), 
                       (data, length, num_ensembles, up_extrema, low_extrema))
        
        # Check if enough extrema exist
        up_count = up_extrema.sum(axis=1)
        low_count = low_extrema.sum(axis=1)
        
        valid_mask = (up_count >= 1) & (low_count >= 1)
        return up_extrema, low_extrema, valid_mask

    def compute_spline(self, data, extrema, type_flag):
        num_ensembles, length = data.shape
        result = cp.empty_like(data, dtype=cp.float32)
        
        block_x = 256
        grid_x = (num_ensembles + block_x - 1) // block_x
        
        spline_kernel((grid_x,), (block_x,),
                      (data, extrema, length, num_ensembles, result, int(type_flag),
                       self.tmp_x, self.tmp_y, self.tmp_a, self.tmp_b, self.tmp_c, self.tmp_rhs))
        
        return result
