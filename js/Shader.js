var Shaders = {};

Shaders["Test"] = `#version 300 es

precision highp float;
precision highp int;
precision highp sampler3D;

uniform sampler3D prev_activation;

layout(location = 0) in float idx_f;

out float z;
out float activation;

void main() {
    uint idx = uint(idx_f);

    uint Z = idx / uint(3 * 4 * 4);
    idx %= uint(3 * 4 * 4);

    uint y = idx / uint(4 * 4);
    idx %= uint(4 * 4);

    uint x = idx / uint(4);
    idx %= uint(4);

    vec4  txl = texelFetch(prev_activation, ivec3(x, y, Z), 0);

    z = idx_f;
    activation = txl[idx] + 0.1;
}`;


Shaders["ConvolutionalLayer-forward"] = `#version 300 es

precision highp float;
precision highp int;
precision highp sampler3D;

uniform float weights[filterCount * filterSize * filterSize];
uniform float biases[filterCount];

uniform sampler3D prev_activation;

layout(location = 0) in float idx_f;

out float z;
out float activation;

void main() {
    uint idx = uint(idx_f);

    uint batch_idx = idx / unitSize;
    idx -= batch_idx * unitSize;

    uint r1 = idx / imgCols_filterCount;
    idx -= r1 * imgCols_filterCount;

    uint c1 = idx / filterCount;
    idx -= c1 * filterCount;

    uint filter_idx = idx;

    uint weight_idx = filter_idx * filterSize_filterSize;

    uint r2;
    float sum = 0.0;
    for (r2 = 0u; r2 < filterSize; r2++) {
        uint x = c1 >> 2u;  // c1 / 4u
        uint u = c1 &  3u;   // c1 % 4u

        uint c2 = 0u;
        for(; ; ) {
            vec4  txl = texelFetch(prev_activation, ivec3(x, r1 + r2, batch_idx), 0);
            for(; u < 4u && c2 < filterSize; u++, c2++) {

                sum += txl[u] * weights[weight_idx];
                weight_idx++;
            }

            if(filterSize <= c2) {
                break;
            }

            x++;
            u = 0u;
        }
    }

    z = sum + biases[filter_idx];
    activation = 1.0 / (1.0 + exp(-z));
}`;


Shaders["vs-Texture"] = `#version 300 es

precision highp float;
precision highp int;

uniform int B_Cols;

uniform sampler2D A_Tex;
uniform sampler2D B_Tex;

layout(location = 0) in float idx_f;

out float dot_val;

void main() {
    uint idx = uint(idx_f);
    int i   = int(idx / uint(B_Cols));
    int j   = int(idx % uint(B_Cols));

    int k;
    float sum = 0.0;
    for(k = 0; k < _repeat_; k++) {
        vec4  A_txl, B_txl;

        A_txl = texelFetch(A_Tex, ivec2(k, i), 0);
        B_txl = texelFetch(B_Tex, ivec2(k, j), 0);
        sum   += dot(A_txl, B_txl);
    }

    dot_val = sum;
}`;

Shaders["vs-Uniform"] = `#version 300 es

precision highp float;
precision highp int;

uniform int B_Cols;

uniform vec4 A[_A_len_];
uniform vec4 B[_B_len_];

layout(location = 0) in float idx_f;

out float dot_val;

void main() {
    uint idx = uint(idx_f);
    int i   = int(idx / uint(B_Cols));
    int j   = int(idx % uint(B_Cols));

    int k;
    float sum = 0.0;
    for(k = 0; k < _repeat_; k++) {
        sum += dot(A[_repeat_*i +k], B[_repeat_*j +k]);
    }
    dot_val = sum;
}`;

Shaders["fs-transform"] = `#version 300 es
precision highp float;
precision highp int;

out vec4 color;

void main(){
    color = vec4(1.0);
}`;
