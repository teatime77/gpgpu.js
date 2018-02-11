
function CreateNeuralNetworkShaders() {

    return {
FullyConnectedLayer_Forward: 
                `in float zero;

            const int ActivationFunction_none       = 0;
            const int ActivationFunction_sigmoid    = 1;
            const int ActivationFunction_ReLU       = 2;

            uniform int activationFunction;

            // 2次元配列のテクスチャ
            uniform sampler2D W;
            uniform sampler2D X;
            uniform sampler2D Bias;

            out float z;
            out float y;

            float sigmoid(float x){
                return 1.0 / (1.0 + exp(-x));
            }

            void main() {
                ivec2 X_sz = textureSize(X, 0);
                ivec2 Bias_sz = textureSize(Bias, 0);

                int batch_idx = gl_VertexID / Bias_sz.x;
                int out_idx   = gl_VertexID % Bias_sz.x;

                float sum = 0.0f;
                for(int i = 0; i < X_sz.x; i++) {

                    vec4 w = texelFetch(W, ivec2(i, out_idx), 0);

                    vec4 x = texelFetch(X, ivec2(i, batch_idx), 0);

                    sum += w.r * x.r;
                }

                vec4 bias = texelFetch(Bias, ivec2(out_idx, 0), 0);
                sum += bias.r;

                // 入力変数zeroの値は必要ないですが、使用しない変数はコンパイラが除去してしまいエラーになるので形の上だけ使用します。
                // zeroの値は0なので計算結果には影響しません。
                z = sum + zero;

                switch(activationFunction){
                case ActivationFunction_none:
                    y = z;
                    break;

                case ActivationFunction_sigmoid:
                    y = sigmoid(z);
                    break;

                case ActivationFunction_ReLU:
                    y = (0.0f < z ? z : 0.0f);
                    break;
                }
            }`
,

FullyConnectedLayer_DeltaWeight:
                `in float zero;

            // 2次元配列のテクスチャ
            uniform sampler2D prev_y;
            uniform sampler2D deltaZ;

            out float deltaWeight;

            void main() {
                int row = gl_VertexID / WeightColSize;
                int col = gl_VertexID % WeightColSize;

                float sum = 0.0f;
                int batch_idx;
                for(batch_idx = 0; batch_idx < miniBatchSize; batch_idx++){

                    vec4 pa = texelFetch(prev_y, ivec2(col, batch_idx), 0);

                    vec4 dz = texelFetch(deltaZ, ivec2(row, batch_idx), 0);

                    sum += pa.r * dz.r;
                }

                deltaWeight = sum + zero;
            }`
,

FullyConnectedLayer_DeltaX:
                `in float zero;

            // 2次元配列のテクスチャ
            uniform sampler2D W;
            uniform sampler2D deltaZ;

            out float deltaX;

            void main() {
                ivec2 W_sz = textureSize(W, 0);

                int batch_idx = gl_VertexID / W_sz.x;
                int delta_x_idx   = gl_VertexID % W_sz.x;

                float sum = 0.0f;
                for(int i = 0; i < W_sz.y; i++) {

                    vec4 w = texelFetch(W, ivec2(delta_x_idx, i), 0);

                    vec4 z = texelFetch(deltaZ, ivec2(i, batch_idx), 0);

                    sum += w.r * z.r;
                }

                deltaX = sum + zero;
            }`
,

ConvolutionalLayer_Forward : `
precision highp sampler3D;

const int ActivationFunction_none       = 0;
const int ActivationFunction_sigmoid    = 1;
const int ActivationFunction_ReLU       = 2;

uniform int activationFunction;

uniform float bias[numChannels];

uniform sampler3D weight;
uniform sampler3D prev_y;

in float zero;

out float z;
out float y;

float sigmoid(float x){
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint idx = uint(gl_VertexID);

    uint batch_idx = idx / (numChannels * numRows * numCols);
    idx     -= batch_idx * (numChannels * numRows * numCols);

    uint channel_idx  = idx / (numRows * numCols);
    idx      -= channel_idx * (numRows * numCols);

    uint r1 = idx / numCols;
    uint c1 = idx - r1 * numCols;

    uint batch_channel_idx = batch_idx * prevNumChannels;

    uint prev_channel_idx, r2, c2;
    float sum = 0.0f;
    uint weight_idx = channel_idx * prevNumChannels;

    for(prev_channel_idx = 0u; prev_channel_idx < prevNumChannels; prev_channel_idx++) {

        for (r2 = 0u; r2 < filterSize; r2++) {

            for (c2 = 0u; c2 < filterSize; c2++) {

                uint c3 = c1 + c2;
                uint r3 = r1 + r2;

                vec4  txl = texelFetch(prev_y, ivec3(c3, r3, batch_channel_idx), 0);

                vec4  w   = texelFetch(weight, ivec3(c2, r2, weight_idx), 0);

                sum += txl.r * w.r;
            }
        }
        batch_channel_idx++;
        weight_idx++;
    }

    z = sum + bias[channel_idx] + zero;

    switch(activationFunction) {
    case ActivationFunction_none:
        y = z;
        break;

    case ActivationFunction_sigmoid:
        y = sigmoid(z);
        break;

    case ActivationFunction_ReLU:
        y = (0.0f < z ? z: 0.0f);
        break;
    }
}`
,

ConvolutionalLayer_DeltaWeights: `
precision highp sampler3D;

uniform sampler3D delta_z;
uniform sampler3D prev_y;

in float zero;

out float delta_w;

void main() {
    uint idx = uint(gl_VertexID);

    uint channel_idx = idx / (prevNumChannels * filterSize * filterSize);
    idx     -= channel_idx * (prevNumChannels * filterSize * filterSize);

    uint prev_channel_idx = idx / (filterSize * filterSize);
    idx     -= prev_channel_idx * (filterSize * filterSize);

    uint r2 = idx / filterSize;
    uint c2 = idx - r2 * filterSize;

    uint r1, c1, batch_idx;
    float sum = 0.0f;

    for (r1 = 0u; r1 < numRows; r1++) {
        uint r3 = r1 + r2;

        for (c1 = 0u; c1 < numCols; c1++) {
            uint c3 = c1 + c2;

            for(batch_idx = 0u; batch_idx < miniBatchSize; batch_idx++) {

                uint this_batch_channel = batch_idx *     numChannels +      channel_idx;
                uint prev_batch_channel = batch_idx * prevNumChannels + prev_channel_idx;

                vec4  dz = texelFetch(delta_z, ivec3(c1, r1, this_batch_channel), 0);

                vec4  pa = texelFetch(prev_y, ivec3(c3, r3, prev_batch_channel), 0);

                sum += dz.r * pa.r;
            }
        }
    }

    delta_w = sum + zero;
}`
,

ConvolutionalLayer_DeltaX : `
precision highp sampler3D;

uniform sampler3D delta_z;
uniform sampler3D weight;

in float zero;

out float delta_x;

void main() {
    uint idx = uint(gl_VertexID);

    uint batch_idx = idx / (prevNumChannels * prevNumRows * prevNumCols);
    idx     -= batch_idx * (prevNumChannels * prevNumRows * prevNumCols);

    uint prev_channel_idx = idx / (prevNumRows * prevNumCols);
    idx     -= prev_channel_idx * (prevNumRows * prevNumCols);

    uint r3 = idx / prevNumCols;
    uint c3 = idx - r3 * prevNumCols;

    uint channel_idx, r2, c2;
    float sum = 0.0f;

    // 出力のチャネルに対し
    for(channel_idx = 0u; channel_idx < numChannels; channel_idx++) {

        uint this_batch_channel = batch_idx * numChannels + channel_idx;
        uint weight_idx = channel_idx * prevNumChannels + prev_channel_idx;

        // フィルターの行に対し
        for (r2 = 0u; r2 < filterSize; r2++) {

            // 出力の行
            uint r1 = r3 - r2;

            if(0u <= r1 && r1 < numRows) {

                // フィルターの列に対し
                for (c2 = 0u; c2 < filterSize; c2++) {

                    // 出力の列
                    uint c1 = c3 -c2;

                    if(0u <= c1 && c1 < numCols) {

                        vec4  dz = texelFetch(delta_z, ivec3(c1, r1, this_batch_channel), 0);

                        vec4  w   = texelFetch(weight, ivec3(c2, r2, weight_idx), 0);

                        sum += dz.r * w.r;
                    }
                }
            }
        }
    }

    delta_x = sum + zero;
}`

};
}