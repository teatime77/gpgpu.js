var Shaders = {};

Shaders["AttribTest"] = `
precision highp sampler3D;

in    float    w;
in vec4      x;
in vec4 y;

uniform float biases[4];
uniform float f;
uniform vec4 ff;
uniform sampler3D tt;

out vec4 z;

void main() {
    uint idx = uint(w) % 4u;

    vec4  txl = texelFetch(tt, ivec3(2, 3, 1), 0);
    z = (x +y +ff +txl) +biases[idx]+f;
}`;


Shaders["Test"] = `
precision highp sampler3D;

in float idx_f;

uniform float biases[4];

uniform sampler3D prev_activation;

out float z;
out float activation;

void main() {
    uint idx = uint(idx_f);

    uint Z = idx / uint(4 * 3 * 28);
    idx %= uint(4 * 3 * 28);

    uint y = idx / uint(4 * 3);
    idx %= uint(4 * 3);

    uint x = idx / uint(4);
    idx %= uint(4);

    vec4  txl = texelFetch(prev_activation, ivec3(x, y, Z), 0);

    z = idx_f;
    activation = txl[idx] + biases[idx];
}`;


Shaders["ConvolutionalLayer-forward"] = `
precision highp sampler3D;

//uniform float weights[channelSize * prevChannelSize * filterSize * filterSize];
uniform float biases[channelSize];

uniform sampler3D weights;
uniform sampler3D prev_activation;

in float zero;

out float z;
out float activation;

void main() {
    uint idx = uint(gl_VertexID);

    uint batch_idx = idx / (channelSize * rowCount * colCount);
    idx     -= batch_idx * (channelSize * rowCount * colCount);

    uint channel_idx  = idx / (rowCount * colCount);
    idx      -= channel_idx * (rowCount * colCount);

    uint r1 = idx / colCount;
    uint c1 = idx - r1 * colCount;

    uint batch_channel_idx = batch_idx * prevChannelSize;

    uint prev_channel_idx, r2, c2;
    float sum = 0.0f;
//    uint weight_idx = channel_idx * prevChannelSize * filterSize * filterSize;
    uint weight_idx = channel_idx * prevChannelSize;

    for(prev_channel_idx = 0u; prev_channel_idx < prevChannelSize; prev_channel_idx++) {

        for (r2 = 0u; r2 < filterSize; r2++) {

            for (c2 = 0u; c2 < filterSize; c2++) {

                uint c3 = c1 + c2;
                uint r3 = r1 + r2;

                vec4  txl = texelFetch(prev_activation, ivec3(c3, r3, batch_channel_idx), 0);

                vec4  w   = texelFetch(weights, ivec3(c2, r2, weight_idx), 0);

                sum += txl.r * w.r;
//                sum += txl.r * weights[weight_idx];
//                weight_idx++;
            }
        }
        batch_channel_idx++;
        weight_idx++;
    }

    z = sum + biases[channel_idx] + zero;
    activation = 1.0 / (1.0 + exp(-z));
}`;


Shaders["ConvolutionalLayer-NablaWeights"] = `
precision highp sampler3D;

uniform sampler3D delta_z;
uniform sampler3D prev_activation;

in float zero;

out float nabla_w;

void main() {
    uint idx = uint(gl_VertexID);

    uint channel_idx = idx / (prevChannelSize * filterSize * filterSize);
    idx     -= channel_idx * (prevChannelSize * filterSize * filterSize);

    uint prev_channel_idx = idx / (filterSize * filterSize);
    idx     -= prev_channel_idx * (filterSize * filterSize);

    uint r2 = idx / filterSize;
    uint c2 = idx - r2 * filterSize;

    uint r1, c1, batch_idx;
    float sum = 0.0f;

    for (r1 = 0u; r1 < rowCount; r1++) {
        uint r3 = r1 + r2;

        for (c1 = 0u; c1 < colCount; c1++) {
            uint c3 = c1 + c2;

            for(batch_idx = 0u; batch_idx < miniBatchSize; batch_idx++) {

                uint this_batch_channel = batch_idx *     channelSize +      channel_idx;
                uint prev_batch_channel = batch_idx * prevChannelSize + prev_channel_idx;

                vec4  dz = texelFetch(delta_z, ivec3(c1, r1, this_batch_channel), 0);

                vec4  pa = texelFetch(prev_activation, ivec3(c3, r3, prev_batch_channel), 0);

                sum += dz.r * pa.r;
            }
        }
    }

    nabla_w = sum + zero;
}`;


Shaders["vs-Texture"] = `
uniform int B_Cols;

uniform sampler2D A_Tex;
uniform sampler2D B_Tex;

in float idx_f;

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

Shaders["vs-Uniform"] =
`
uniform int B_Cols;

uniform vec4 A[_A_len_];
uniform vec4 B[_B_len_];

in float idx_f;

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
