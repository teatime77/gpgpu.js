﻿
var miniBatchSize;
var learningRate;
var useSoftMax = true;
var WebGL2;
var isTest = false;

function Stats(tm, idx){
    switch(tm.length){
    case 0:
        return "-"
    case 1:
        return "" + Math.floor(tm[0] / idx)
    default:
        return "[" + tm.map(x => Math.floor(x / idx)).reduce((x,y) => x + "," + y) + "]";
    }
}

class Lap {
    constructor(v){
        this.lastTime = new Date();
        this.lapIdx = 0;
        this.lapTimes = v;
    }

    Time(){
        var prev_last_time = this.lastTime;
        this.lastTime = new Date();

        if(this.lapTimes.length <= this.lapIdx){
            this.lapTimes.push(0);
        }
        this.lapTimes[this.lapIdx] += this.lastTime - prev_last_time;
        this.lapIdx++;
    }
}

class Layer {
    constructor() {
    }

    init(prev_layer) {
        this.prevLayer = prev_layer;
        if (prev_layer) {
            prev_layer.nextLayer = this;
        }
    }

    miniBatchSizeChanged(){
        this.fwTime = [];
        this.bwTime = [];
        this.udTime = [];
    }

    forward() {
    }

    backward() {
    }

    updateParameter() {
    }

    clear(){
        if(this.params){

            for(var key in this.params){
                WebGL2.clear(this.params[key].id);
            }
            this.params = {};
        }
    }
}

class InputLayer extends Layer {
    constructor(rows, cols) {
        super();

        this.imgRows = rows;
        this.imgCols = cols;
        this.unitSize = rows * cols;
    }
}

class FullyConnectedLayer extends Layer{
    constructor(size) {
        super();

        this.unitSize = size;
        this.params = {};
    }

    init(prev_layer) {
        super.init(prev_layer);

        this.bias = np.random.randn(this.unitSize, 1);
        this.weight = np.random.randn(this.unitSize, this.prevLayer.unitSize);
    }

    miniBatchSizeChanged(){
        super.miniBatchSizeChanged();

        this.outZero    = new Float32Array(miniBatchSize * this.unitSize);
        this.z          = new ArrayView(miniBatchSize,  this.unitSize);
        this.activation = new ArrayView(miniBatchSize,  this.unitSize);
        this.deltaX     = new ArrayView(miniBatchSize,  this.prevLayer.unitSize);
        this.prevLayerActivation = new ArrayView(miniBatchSize,  this.prevLayer.unitSize);

        if(!this.nextLayer){
            // 最後の場合

            this.costDerivative = new ArrayView(miniBatchSize,  this.unitSize);
        }
    }

    gpuForwardSigmoid(){
        var vertex_shader =
            `in float zero;

        // 2次元配列のテクスチャ
        uniform sampler2D W;
        uniform sampler2D X;
        uniform sampler2D Bias;

        out float z;
        out float activation;

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

            activation = sigmoid(z);
        }`;

        /*
        var param_id = "Fully-Connected-Layer-forward," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize;
        if (true || this.params[param_id] == undefined){

            this.params[param_id] = {
                id : param_id,
                vertexShader: vertex_shader,
                args : {
                    "zero": this.outZero,
                    "X": WebGL2.makeTextureInfo("float", [ miniBatchSize, this.prevLayer.unitSize]),
                    "W": WebGL2.makeTextureInfo("float", this.weight.shape, this.weight.dt),
                    "Bias": WebGL2.makeTextureInfo("float", [ 1, this.bias.dt.length ], this.bias.dt),
                    "z": this.z.dt,
                    "activation" : this.activation.dt
                }
            };
        }

        var param = this.params[param_id];
        //param.args["X"].value = this.prevLayer.activation.dt;
        //param.args["W"].value = this.
        //param.args["Bias"].value = this.

        WebGL2.compute(param);
        */

        this.param = {
            id : "Fully-Connected-Layer-forward," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize,
                    vertexShader: vertex_shader,
                args : {
                "zero": this.outZero,
                "X": WebGL2.makeTextureInfo("float", [ miniBatchSize, this.prevLayer.unitSize], this.prevLayer.activation.dt),
                "W": WebGL2.makeTextureInfo("float", this.weight.shape, this.weight.dt),
                "Bias": WebGL2.makeTextureInfo("float", [ 1, this.bias.dt.length ], this.bias.dt),
                "z": this.z.dt,
                "activation" : this.activation.dt
                }
                };

        WebGL2.compute(this.param);

    }


    gpuForwardSoftMax(){
        var vertex_shader =
            `in float zero;

        // 2次元配列のテクスチャ
        uniform sampler2D W;
        uniform sampler2D X;
        uniform sampler2D Bias;

        out float activation;

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

            activation = sum + zero;
        }`;

        var param_id = "Fully-Connected-Layer-forward-soft-max," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize;
        if (this.params[param_id] == undefined){

            this.params[param_id] = {
                id : param_id,
                vertexShader: vertex_shader,
                args : {
                    "zero": this.outZero,
                    "X": WebGL2.makeTextureInfo("float", [ miniBatchSize, this.prevLayer.unitSize], this.prevLayer.activation.dt),
                    "W": WebGL2.makeTextureInfo("float", this.weight.shape, this.weight.dt),
                    "Bias": WebGL2.makeTextureInfo("float", [ 1, this.bias.dt.length ], this.bias.dt),
                    "activation" : this.activation.dt
                }
            };
        }

        var param = this.params[param_id];
        param.args["X"].value = this.prevLayer.activation.dt;

        WebGL2.compute(param);
    }


    forward() {
        var lap = new Lap(this.fwTime);
        if(useSoftMax && ! this.nextLayer){

            this.gpuForwardSoftMax();
        }
        else{

            this.gpuForwardSigmoid();
        }

        lap.Time();
    }

    gpuDeltaX(){
        var vertex_shader =
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
        }`;

        var param_id = "Fully-Connected-Layer-gpu-delta-X," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize;
        if (this.params[param_id] == undefined){

            this.params[param_id] = {
                id : param_id,
                vertexShader: vertex_shader,
                args : {
                    "zero": new Float32Array(miniBatchSize * this.prevLayer.unitSize),
                    "W": makeTextureInfo(WebGL2, "float", this.weight),
                    "deltaZ": makeTextureInfo(WebGL2, "float", this.deltaZ),
                    "deltaX" : this.deltaX.dt
                }
            };
        }

        var param = this.params[param_id];
        param.args["deltaZ"].value = this.deltaZ.dt;

        WebGL2.compute(param);
    }

    cpuDeltaX(){

        // 出力先
        var output_idx = 0;

        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

            // 入力に対し
            for (var x_idx = 0; x_idx < this.prevLayer.unitSize; x_idx++) {

                var sum = 0.0;

                // 重みの行とδzの内積
                for (var k = 0; k < this.weight.nrow; k++) {
                    var weight_idx = k * this.weight.ncol + x_idx;
                    var delta_z_idx = batch_idx * this.unitSize + k;
                    sum += this.deltaZ.dt[delta_z_idx] * this.weight.dt[weight_idx];
                }

                this.deltaX.dt[output_idx] = sum;
                output_idx++;
            }
        }
    }

    backward() {
        var lap = new Lap(this.bwTime);

        if (! this.nextLayer) {
            // 最後のレイヤーの場合

            if(useSoftMax){

                this.deltaZ = new ArrayView(miniBatchSize, this.unitSize, this.costDerivative.dt);
            }
            else{

                this.deltaZ = this.costDerivative.Mul(sigmoid_prime(this.z));
            }
        }
        else{
            // 最後のレイヤーでない場合

            this.costDerivative = this.nextLayer.deltaX;
            this.deltaZ = this.costDerivative.Mul(sigmoid_prime(this.z));
        }

        lap.Time();

        this.nabla_b = this.deltaZ.Reduce((x, y) => x + y);
        lap.Time();

        if(false){

            this.nabla_w = np.dot(this.deltaZ, this.prevLayer.activation);
            lap.Time();

            //!!!!! 直前が入力層なら必要なし !!!!!
            this.deltaX = np.dot(this.weight.T(), this.deltaZ);
        }
        else{
            this.prevLayerActivation.dt = this.prevLayer.activation.dt;
            this.nabla_w = this.deltaZ.T().Dot2(this.prevLayerActivation);
            lap.Time();

            //!!!!! 直前が入力層なら必要なし !!!!!
            this.gpuDeltaX();
            if(Math.random() < 0.01){

                var gpu_delta_x = new Float32Array(this.deltaX.dt);
                this.cpuDeltaX();

                var diff = this.deltaX.diff(gpu_delta_x);
                Assert(diff < 0.01, "delta-X");
            }
        }
        lap.Time();
    }

    updateParameter() {
        var eta = learningRate / miniBatchSize;

        var lap = new Lap(this.udTime);

        for(var i = 0; i < this.weight.dt.length; i++){
            this.weight.dt[i] -= eta * this.nabla_w.dt[i];
        }
        lap.Time();

        for(var i = 0; i < this.bias.dt.length; i++){
            this.bias.dt[i] -= eta * this.nabla_b.dt[i];
        }
        lap.Time();
    }
}

class ConvolutionalLayer extends Layer{
    constructor(filter_size, feature_count) {
        super();

        this.filterSize = filter_size;
        this.featureCount = feature_count;
        this.params = {};
    }

    init(prev_layer) {
        super.init(prev_layer);

        Assert(this.prevLayer instanceof InputLayer, "Convolutional-Layer-init");

        this.imgRows = this.prevLayer.imgRows - this.filterSize + 1;
        this.imgCols = this.prevLayer.imgCols - this.filterSize + 1;
        this.unitSize = this.featureCount * this.imgRows * this.imgCols;

        // xrange(this.featureCount).map(x => np.random.randn());
        this.biases = new ArrayView(this.featureCount);
        for (var i = 0; i < this.biases.dt.length; i++) {
            this.biases.dt[i] = np.random.randn();
        }

        this.weights = new ArrayView(this.featureCount, this.filterSize, this.filterSize);
        for (var i = 0; i < this.weights.dt.length; i++) {
            this.weights.dt[i] = np.random.randn();
        }
    }

    miniBatchSizeChanged(){
        super.miniBatchSizeChanged();

        this.forwardCnt = 0;
        this.forwardGPU = 0;
        this.forwardCPU = 0;

        this.z = new ArrayView(miniBatchSize, this.unitSize);
        this.activation = new ArrayView(miniBatchSize, this.unitSize);
    }

    gpuForward() {
        var prev_Layer = this.prevLayer;

        var prev_activation = new ArrayView(miniBatchSize, prev_Layer.unitSize, prev_Layer.activation.dt);

        var vs_id = "ConvolutionalLayer-forward";
        var param_id = vs_id + ":" + this.filterSize + ":" + this.featureCount + ":" + this.imgRows + ":" + this.imgCols + ":" + miniBatchSize;

        if (this.params[param_id] == undefined) {

            var shader_src = Shaders[vs_id]
                .replace(/featureCount/g, this.featureCount.toString() + "u")
                .replace(/rowCount/g, this.imgRows.toString() + "u")
                .replace(/colCount/g, this.imgCols.toString() + "u")
                .replace(/filterSize/g, this.filterSize.toString() + "u");

            this.params[param_id]  = {
                id : param_id,
                vertexShader: shader_src,
                args : {
                    "idx_f": MakeFloat32Index(miniBatchSize * this.featureCount * this.imgRows * this.imgCols),
                    "prev_activation": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize, prev_Layer.imgRows, prev_Layer.imgCols)),
                    "weights": this.weights.dt,
                    "biases": this.biases.dt,
                    "z": this.z.dt,
                    "activation": this.activation.dt
                }
            };
        }

        var param = this.params[param_id];

        param.args["prev_activation"].value = prev_activation.dt;
        WebGL2.compute(param);
    }

    cpuForward() {
        var prev_Layer = this.prevLayer;

        var prev_activation_dt = prev_Layer.activation.dt;
        var z_dt = this.z.dt;
        var activation_dt = this.activation.dt;

        // 出力先
        var output_idx = 0;

        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

            // すべての特徴マップに対し
            for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

                // 出力の行に対し
                for (var r1 = 0; r1 < this.imgRows; r1++) {

                    // 出力の列に対し
                    for (var c1 = 0; c1 < this.imgCols; c1++) {

                        var sum = 0.0;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {
                                var weight_idx = (feature_idx * this.filterSize + r2) * this.filterSize + c2;
                                var prev_activation_idx = batch_idx * prev_Layer.unitSize + (r1 + r2) * prev_Layer.imgCols + (c1 + c2);
                                sum += prev_activation_dt[prev_activation_idx] * this.weights.dt[weight_idx];
                            }
                        }

                        var z_val = sum + this.biases.dt[feature_idx];

                        z_dt[output_idx] = z_val;
                        activation_dt[output_idx] = sigmoidF(z_val);

                        output_idx++;
                    }
                }
            }
        }
    }

    forward() {
        var lap = new Lap(this.fwTime);

        var t0 = new Date();
        this.gpuForward();
        var t1 = new Date();

        lap.Time();

        if(this.forwardCnt != 0 && 0.1 < Math.random()){

            return;
        }

        var z_gpu_dt = new Float32Array(this.z.dt);
        var activation_gpu_dt = new Float32Array(this.activation.dt);

        this.cpuForward();
        var t2 = new Date();

        var max_diff = 0;

        // 出力先
        var output_idx = 0;
        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {
            for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {
                for (var r1 = 0; r1 < this.imgRows; r1++) {
                    for (var c1 = 0; c1 < this.imgCols; c1++) {
                        var diff = Math.max(Math.abs(z_gpu_dt[output_idx] - this.z.dt[output_idx]), Math.abs(activation_gpu_dt[output_idx] - this.activation.dt[output_idx]));
                        if (max_diff < diff) {
                            if(0.0001 < diff){
                                console.log("");
                            }
                            max_diff = diff;
                        }
                        output_idx++;
                    }
                }
            }
        }
        Assert(max_diff < 0.0001, "Convolutional-Layer-forward-diff");
        this.forwardCnt++;
        this.forwardGPU += t1 - t0;
        this.forwardCPU += t2 - t1;
        if (this.forwardCnt % 100 == 0 || 100 <= miniBatchSize) {

//            console.log("forward diff:%f cnt:%d GPU:%dms CPU:%dms", max_diff, this.forwardCnt, Math.round(this.forwardGPU / this.forwardCnt), Math.round(this.forwardCPU / this.forwardCnt));
        }
    }

    gpuNablaWeights(delta_z) {
        var prev_Layer = this.prevLayer;

        var prev_activation = new ArrayView(prev_Layer.imgRows, prev_Layer.imgCols, miniBatchSize, prev_Layer.activation.dt);
        var delta_z_3D = new ArrayView(this.featureCount, this.imgRows, this.imgCols * miniBatchSize, delta_z.dt);

        var batch_vec4_count = miniBatchSize / 4;
        var vs_id = "ConvolutionalLayer-backward";
        var param_id = vs_id + ":" + this.featureCount + ":" + this.filterSize + ":" + this.imgRows + ":" + this.imgCols + ":" + miniBatchSize;
        var param;

        if (this.params[param_id] == undefined) {

            param = {};

            param.id = param_id;

            this.params[param.id] = param;

            param.elementCount = this.featureCount * this.filterSize * this.filterSize;

            param.args = {
                "idx_f": MakeFloat32Index(param.elementCount),
                "prev_activation": makeTextureInfo(WebGL2, "vec4", prev_activation),
                "delta_z": makeTextureInfo(WebGL2, "vec4", delta_z_3D),
                "nablaWeights": this.nablaWeights,
            };

            var shader_src = Shaders[vs_id]
                .replace(/featureCount/g, this.featureCount.toString() + "u")
                .replace(/rowCount/g, this.imgRows.toString() + "u")
                .replace(/colCount/g, this.imgCols.toString() + "u")
                .replace(/batchVec4Count/g, batch_vec4_count.toString() + "u")
                .replace(/filterSize/g, this.filterSize.toString() + "u");

            console.log(shader_src);
            param.vertexShader = shader_src;
        }
        else {

            param = this.params[param_id];
        }

        WebGL2.compute(param);
    }

    cpuNablaWeights(delta_z) {
        var prev_Layer = this.prevLayer;

        // すべての特徴マップに対し
        var weights_idx = 0;
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            // フィルターの行に対し
            for (var r2 = 0; r2 < this.filterSize; r2++) {

                // フィルターの列に対し
                for (var c2 = 0; c2 < this.filterSize; c2++) {

                    var nabla_w = 0.0;

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.imgRows; r1++) {

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.imgCols; c1++) {

                            // バッチ内のデータに対し
                            var delta_z_idx = feature_idx * (r1 * c1) + r1 * (this.imgCols | 0);
                            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                                var delta = delta_z.dt[delta_z_idx];
                                if (delta != 0) {

                                    var prev_activation_idx = batch_idx * (prev_Layer.unitSize | 0) + (r1 + r2) * (prev_Layer.imgCols | 0) + (c1 + c2);
                                    nabla_w += delta * prev_Layer.activation.dt[prev_activation_idx];
                                }
                                delta_z_idx += this.unitSize;
                            }
                        }
                    }

                    this.nablaWeights.dt[weights_idx] = nabla_w;
                    weights_idx++;
                }
            }
        }
        Assert(weights_idx == this.nablaWeights.dt.length);
    }

    cpuNablaBiases(delta_z){
        // すべての特徴マップに対し
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            var nabla_b = 0.0;

            // 出力の行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {

                // 出力の列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {

                    // バッチ内のデータに対し
                    var delta_z_idx = feature_idx * (r1 * c1) + r1 * this.imgCols;
                    for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                        nabla_b += delta_z.dt[delta_z_idx];
                        delta_z_idx += this.unitSize;

                        //!!!!!! 直前が入力層なら必要なし !!!!!
                        // this.costDerivative.dt[output_idx] = this.nextLayer.DeltaT[output_idx];
                    }
                }
            }

            this.nablaBiases.dt[feature_idx] = nabla_b;
        }
    }

    backward() {
        var lap = new Lap(this.bwTime);

        var delta_z = new ArrayView(miniBatchSize, this.unitSize, new Float32Array(this.nextLayer.deltaX.dt));
        lap.Time();

        for (var i = 0; i < delta_z.dt.length; i++) {
            delta_z.dt[i] *= sigmoid_primeF(this.z.dt[i]);
        }
        lap.Time();

        this.nablaBiases = new ArrayView(this.featureCount, 1);
        this.nablaWeights = new ArrayView(this.featureCount, this.filterSize, this.filterSize);
        this.costDerivative = new ArrayView(this.unitSize, 1);
        lap.Time();

        this.cpuNablaBiases(delta_z);
        lap.Time();

        //this.gpuNablaWeights(delta_z);
        //var gpu_nabla_weights = new Float32Array(this.nablaWeights.dt);
        //lap.Time();

        this.cpuNablaWeights(delta_z);
//        AssertEq(gpu_nabla_weights, this.nablaWeights.dt);
        lap.Time();
    }

    updateParameter() {
        var lap = new Lap(this.udTime);

        var eta = learningRate / miniBatchSize;

        // すべての特徴マップに対し
        var weights_idx = 0;
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            this.biases.dt[feature_idx] -= eta * this.nablaBiases.dt[feature_idx];

            // フィルターの行に対し
            for (var r2 = 0; r2 < this.filterSize; r2++) {

                // フィルターの列に対し
                for (var c2 = 0; c2 < this.filterSize; c2++) {
                    this.weights.dt[weights_idx] -= eta * this.nablaWeights.dt[weights_idx];
                    weights_idx++;
                }
            }
        }
        Assert(weights_idx == this.weights.dt.length);
        lap.Time();
    }

    clear(){
        for(var key in this.params){
            WebGL2.clear(this.params[key].id);
        }
        this.params = {};
    }
}

class PoolingLayer extends Layer {
    constructor(filter_size) {
        super();
        this.filterSize = filter_size;
    }

    init(prev_layer) {
        super.init(prev_layer);

        Assert(this.prevLayer instanceof ConvolutionalLayer, "Pooling-Layer-init");

        this.featureCount = this.prevLayer.featureCount;
        this.imgRows = this.prevLayer.imgRows / this.filterSize;
        this.imgCols = this.prevLayer.imgCols / this.filterSize;

        Assert(Math.ceil(this.imgRows) == this.imgRows && Math.ceil(this.imgCols) == this.imgCols);

        this.unitSize = this.featureCount * this.imgRows * this.imgCols;
    }

    miniBatchSizeChanged(){
        super.miniBatchSizeChanged();

        this.activation = new ArrayView(miniBatchSize, this.unitSize);
        this.maxRow     = new Int8Array(miniBatchSize, this.unitSize);
        this.maxCol     = new Int8Array(miniBatchSize, this.unitSize);
        this.deltaX = new ArrayView(miniBatchSize, this.prevLayer.unitSize);
    }

    forward() {
        var lap = new Lap(this.fwTime);

        var prev_Layer = this.prevLayer;

        var prev_activation_dt = prev_Layer.activation.dt;

        // 出力先
        var output_idx = 0;

        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

            // すべての特徴マップに対し
            for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

                // 出力の行に対し
                for (var r1 = 0; r1 < this.imgRows; r1++) {
                    var r0 = r1 * this.filterSize;

                    // 出力の列に対し
                    for (var c1 = 0; c1 < this.imgCols; c1++) {
                        var c0 = c1 * this.filterSize;

                        var max_val = -10000;
                        var max_row, max_col;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {

                                var prev_activation_idx = batch_idx * prev_Layer.unitSize + (feature_idx * prev_Layer.imgRows + (r0 + r2)) * prev_Layer.imgCols + (c0 + c2);
                                var val = prev_activation_dt[prev_activation_idx];
                                if (max_val < val) {

                                    max_val = val;
                                    max_row = r2;
                                    max_col = c2;
                                }
                            }
                        }

                        this.activation.dt[output_idx] = max_val;
                        this.maxRow[output_idx] = max_row;
                        this.maxCol[output_idx] = max_col;

                        output_idx++;
                    }
                }
            }
        }

        Assert(output_idx == this.activation.dt.length);
        lap.Time();
    }

    backward() {
        var lap = new Lap(this.bwTime);

        var prev_Layer = this.prevLayer;

        Assert(this.activation.dt.length == this.nextLayer.deltaX.dt.length);

        for(var i = 0; i < this.deltaX.dt.length; i++){
            this.deltaX.dt[i] = 0;
        }
        lap.Time();

        // 出力先
        var output_idx = 0;

        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

            // すべての特徴マップに対し
            for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {
                var offset = batch_idx * prev_Layer.unitSize + feature_idx * prev_Layer.imgRows * prev_Layer.imgCols;

                // 出力の行に対し
                for (var r1 = 0; r1 < this.imgRows; r1++) {
                    var r0 = r1 * this.filterSize;

                    // 出力の列に対し
                    for (var c1 = 0; c1 < this.imgCols; c1++) {
                        var c0 = c1 * this.filterSize;

                        var delta = this.nextLayer.deltaX.dt[output_idx];
                        if(delta != 0){

                            var r2 = this.maxRow[output_idx] | 0;
                            var c2 = this.maxCol[output_idx] | 0;
                            var prev_activation_idx = offset + (r0 + r2) * prev_Layer.imgCols + (c0 + c2);
                            this.deltaX.dt[prev_activation_idx] = delta;
                        }

                        output_idx++;
                    }
                }
            }
        }

        Assert(output_idx == this.nextLayer.deltaX.dt.length);
        lap.Time();
    }
}


class DropoutLayer extends Layer {
    constructor(ratio) {
        super();
        this.ratio = ratio;
    }

    init(prev_layer) {
        super.init(prev_layer);
        this.unitSize = prev_layer.unitSize;
    }


    miniBatchSizeChanged(){
        super.miniBatchSizeChanged();

        this.activation = new ArrayView(miniBatchSize, this.unitSize);
        this.deltaX     = new ArrayView(miniBatchSize, this.unitSize);
        this.valid      = new Int8Array(miniBatchSize * this.unitSize);
    }

    forward() {
        var lap = new Lap(this.fwTime);
        for(var i = 0; i < this.activation.dt.length; i++){
            if(! isTest){

                if(this.ratio <= Math.random()){

                    this.valid[i]   = 1;
                    this.activation.dt[i] = this.prevLayer.activation.dt[i];
                }
                else{

                    this.valid[i]   = 0;
                    this.activation.dt[i] = 0;
                }
            }
            else{

                this.activation.dt[i] = (1 - this.ratio) *  this.prevLayer.activation.dt[i];
            }
        }
        lap.Time();
    }

    backward() {
        var lap = new Lap(this.bwTime);
        for(var i = 0; i < this.activation.dt.length; i++){
            if(this.valid[i] == 1){

                this.deltaX.dt[i] = this.nextLayer.deltaX.dt[i];
            }
            else{

                this.deltaX.dt[i] = 0;
            }
        }
        lap.Time();
    }
}

class NeuralNetwork {
    constructor(gpgpu, layers) {
        WebGL2 = gpgpu;
        this.layers = layers;
        this.lastLayer = layers[layers.length - 1];

        var prev_layer = null;
        for(let layer of layers) {
            layer.init(prev_layer);
            prev_layer = layer;
        }
    }

    Laminate(data, idx_list, idx_start, idx_cnt) {
        var element_size = data.shape.slice(1).reduce((x, y) => x * y);

        var shape = data.shape.slice();
        shape[0] = idx_cnt;
        var X = new ArrayView(shape);
        var dst = 0;
        for (var idx = idx_start; idx < idx_start + idx_cnt; idx++) {
            var src = idx_list[idx] * element_size;
            for (var i = 0; i < element_size; i++) {
                X.dt[dst] = data.dt[src];
                src++;
                dst++;
            }
        }

        return X;
    }

    costAvg(cost) {
        return xrange(cost.ncol).map(c => cost.Col(c).dt.map(x => Math.abs(x)).reduce((x, y) => x + y) / cost.nrow);
    }

    feedforward(a) {
        for(let l of this.layers.slice(1)) {
            if (l instanceof FullyConnectedLayer) {

                a = sigmoid(np.dot(l.weight, a).Add(l.bias));
            }
        }

        return a;
    }

    CorrectCount(Y){
        var result = this.lastLayer.activation;

        var ok_cnt = 0;
        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {
            var max_idx = np.argmax(result.Row(batch_idx));
            if(Y.dt[batch_idx * this.lastLayer.unitSize + max_idx] == 1){
                ok_cnt++;
            }
        }

        return ok_cnt;
    }

    /*
    損失関数の微分
    */
    SoftMax(cost_derivative, last_y, batch_Y, exp_work, range_len) {
        var cost_sum = 0;
        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++){

            var max_val = -10000;
            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                if (max_val < last_y[k]) {
                    max_val = last_y[k];
                }
            }

            var sum = 0;
            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                var d = Math.exp(last_y[k] - max_val);
                sum += d;
                exp_work[i] = d;
            }

            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                var y = exp_work[i] / sum;
                cost_derivative[k] = y - batch_Y[k];

                cost_sum += (batch_Y[k] * Math.log(y));
            }
        }

        return - cost_sum / miniBatchSize;
    }

    * SGD(training_data, test_data, epochs, mini_batch_size, learning_rate) {
        learningRate = learning_rate;
        var last_layer = this.layers[this.layers.length - 1];
        last_layer.costDerivative = new ArrayView(mini_batch_size, last_layer.unitSize);
        var exp_work = new Float32Array(last_layer.unitSize);

        for (let epoch_idx of xrange(epochs)) {

            for(var mode = 0; mode < 2; mode++){
                var data;
                var ok_cnt = 0;

                isTest = (mode == 1);
                if(mode == 0){

                    data = training_data;
                    miniBatchSize = mini_batch_size;
                }
                else{

                    data = test_data;
                    miniBatchSize = 100 * mini_batch_size;
                }

                this.layers.forEach(x => x.miniBatchSizeChanged());

                var idx_list = np.random.RandomSampling(data.X.shape[0]);

                var show_time = new Date();

                var data_cnt = data.X.shape[0];
                var mini_batch_cnt = Math.floor(data_cnt / miniBatchSize);
                var mini_batch_time = [];
                for (var idx = 0; idx < mini_batch_cnt; idx++) {
                    var lap = new Lap(mini_batch_time);
                    var X = this.Laminate(data.X, idx_list, idx * miniBatchSize, miniBatchSize);
                    lap.Time();
                    var Y = this.Laminate(data.Y, idx_list, idx * miniBatchSize, miniBatchSize);
                    lap.Time();

                    this.layers[0].activation = X;
                    this.layers.forEach(x => x.forward());
                    lap.Time();

                    if(mode == 0){

                        if(useSoftMax){

                            var cost = this.SoftMax(last_layer.costDerivative.dt, last_layer.activation.dt, Y.dt, exp_work, last_layer.unitSize);
                        }
                        else{

                            last_layer.costDerivative = cost_derivative(last_layer.activation, Y);                    
                        }
                        lap.Time();

                        for (var i = this.layers.length - 1; 1 <= i; i--) {
                            this.layers[i].backward();
                        }
                        lap.Time();

                        this.layers.forEach(x => x.updateParameter());
                        lap.Time();
                    }

                    ok_cnt += this.CorrectCount(Y);

                    if (60 * 1000 < new Date() - show_time) {

                        var s = Stats(mini_batch_time, idx);
                        for(let layer of this.layers.slice(1)) {
                            s += " (" + Stats(layer.fwTime, idx) + " " + Stats(layer.bwTime, idx) + " " + Stats(layer.udTime, idx) + ")";
                        }
                        console.log("update mini batch: %.2f %d  %s", ok_cnt / (idx * miniBatchSize), idx * miniBatchSize, s);
                        yield 1;

                        show_time = new Date();
                    }
                }

                this.layers.forEach(x => x.clear());

                if(mode == 1){

                    console.log("Epoch %d  %d / %d eta:%.02f", epoch_idx, ok_cnt, mini_batch_cnt * miniBatchSize, learningRate);
                }
            }

            yield 2;
        }

        yield 0;
    }
}

function cost_derivative(output_activations, y){
    return (output_activations.Sub(y));
}

function sigmoid(z){
//??    return 1.0 / (1.0 + np.exp(-z));
    return z.Map(x => sigmoidF(x));
}

function sigmoid_prime(z){
//??    return sigmoid(z) * (1 - sigmoid(z));
    return z.Map(x => sigmoid_primeF(x));
}

function sigmoid_primeF(z) {
    var f = sigmoidF(z);
    return f * (1 - f);
}

//??
function sigmoidF(z){
    return 1.0 / (1.0 + Math.exp(-z));
}
