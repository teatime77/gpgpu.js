var miniBatchSize;

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

    backward(Y, eta2) {
    }

    updateParameter(eta2) {
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
    }

    init(prev_layer) {
        super.init(prev_layer);

        this.bias = np.random.randn(this.unitSize, 1);
        this.weight = np.random.randn(this.unitSize, this.prevLayer.unitSize);
    }

    miniBatchSizeChanged(){
        this.outZero = new Float32Array(miniBatchSize * this.unitSize);
        this.z2 = new Float32Array(miniBatchSize * this.unitSize);
        this.activation2 = new Float32Array(miniBatchSize * this.unitSize);
    }

    gpuForward(){
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

        var param = {
            id : "Fully-Connected-Layer-forward," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize,
            vertexShader: vertex_shader,
            args : {
                "zero": this.outZero,
                "X": WebGL2.makeTextureInfo("float", [ miniBatchSize, this.prevLayer.unitSize], this.prevLayer.activation.T().dt),
                "W": WebGL2.makeTextureInfo("float", this.weight.shape, this.weight.dt),
                "Bias": WebGL2.makeTextureInfo("float", [ 1, this.bias.dt.length ], this.bias.dt),
                "z": this.z2,
                "activation" : this.activation2
            }
        };

        WebGL2.compute(param);

        this.z = (new ArrayView(miniBatchSize,  this.unitSize, this.z2)).T();
        this.activation = (new ArrayView(miniBatchSize,  this.unitSize, this.activation2)).T();

//        var z_diff = this.z.diff(this.z3);
//        var a_diff = this.activation.diff(this.activation3);
    }

    forward() {
        var lap = new Lap(this.fwTime);

        if(false){

            this.z = np.dot(this.weight, this.prevLayer.activation).AddVec(this.bias);
            lap.Time();

            this.activation = sigmoid(this.z);
        }
        else{

            this.gpuForward();
        }
        lap.Time();
    }

    backward(Y, eta2) {
        var lap = new Lap(this.bwTime);

        if (!this.nextLayer) {
            // 最後のレイヤーの場合

            this.costDerivative = cost_derivative(this.activation, Y);
        }
        else {
            // 最後のレイヤーでない場合

            this.costDerivative = this.nextLayer.deltaX;
        }

        this.Delta = this.costDerivative.Mul(sigmoid_prime(this.z));
        lap.Time();

        this.nabla_b = this.Delta.Reduce((x, y) => x + y);
        lap.Time();

        if(false){

            this.nabla_w = np.dot(this.Delta, this.prevLayer.activation.T());
            lap.Time();

            //!!!!! 直前が入力層なら必要なし !!!!!
            this.deltaX = np.dot(this.weight.T(), this.Delta);
        }
        else{

            this.nabla_w = this.Delta.Dot2(this.prevLayer.activation.T());
//            var diff1 = this.nabla_w2.diff(this.nabla_w);
            lap.Time();

            //!!!!! 直前が入力層なら必要なし !!!!!
            this.deltaX = this.weight.T().Dot2(this.Delta);
//            var diff2 = this.deltaX2.diff(this.deltaX);
        }
        lap.Time();
    }

    updateParameter(eta2) {

//        this.weight = this.weight.Sub(eta2.Mul(this.nabla_w));
//        this.bias = this.bias.Sub(eta2.Mul(this.nabla_b));

        var lap = new Lap(this.udTime);

        for(var i = 0; i < this.weight.dt.length; i++){
            this.weight.dt[i] -= eta2 * this.nabla_w.dt[i];
        }
        lap.Time();

        for(var i = 0; i < this.bias.dt.length; i++){
            this.bias.dt[i] -= eta2 * this.nabla_b.dt[i];
        }
        lap.Time();
    }
}

class ConvolutionalLayer extends Layer{
    constructor(filter_size, feature_count) {
        super();

        this.filterSize = filter_size;
        this.featureCount = feature_count;
        this.param = {};
    }

    init(prev_layer) {
        super.init(prev_layer);

        Assert(this.prevLayer instanceof InputLayer, "Convolutional-Layer-init");

        this.imgRows = this.prevLayer.imgRows - this.filterSize + 1;
        this.imgCols = this.prevLayer.imgCols - this.filterSize + 1;
        this.unitSize = this.imgRows * this.imgCols * this.featureCount;

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
        this.forwardCnt = 0;
        this.forwardGPU = 0;
        this.forwardCPU = 0;

        this.z = new ArrayView(this.unitSize, miniBatchSize);
        this.activation = new ArrayView(this.unitSize, miniBatchSize);
    }

    gpuForward() {
        var prev_Layer = this.prevLayer;

        var sub_batch_size;

        if (miniBatchSize == 12) {

            sub_batch_size = miniBatchSize;
        }
        else if (miniBatchSize == 10000) {
            sub_batch_size = 200;// 16;
        }
        else {

            Assert(false);
        }

        for(var i = 0; i < prev_Layer.activation.dt.length; i++){
            prev_Layer.activation.dt[i] = 0.01 * (i % 100);
        }

        var prev_activation = new ArrayView(prev_Layer.unitSize, miniBatchSize, prev_Layer.activation.dt);

        var vs_id = "ConvolutionalLayer-forward";
        var param_id = vs_id + ":" + this.filterSize + ":" + this.featureCount + ":" + this.imgRows + ":" + this.imgCols + ":" + sub_batch_size;
        var param;

        if (this.param[param_id] == undefined) {

            param = {};

            param.id = param_id;

            this.param[param.id] = param;

            param.sub_prev_activation = new ArrayView(sub_batch_size, prev_Layer.imgRows, prev_Layer.imgCols);
            param.sub_z = new ArrayView(sub_batch_size, this.unitSize);
            param.sub_activation = new ArrayView(sub_batch_size, this.unitSize);

            param.elementCount = sub_batch_size * this.featureCount * this.imgRows * this.imgCols;

            param.args = {
                "idx_f": MakeFloat32Index(param.elementCount),
                "prev_activation": makeTextureInfo(WebGL2, "float", param.sub_prev_activation),
                "weights": this.weights.dt,
                "biases": this.biases.dt,
                "z": param.sub_z.dt,
                "activation": param.sub_activation.dt
            };

            var shader_src = Shaders[vs_id]
                .replace(/featureCount/g, this.featureCount.toString() + "u")
                .replace(/rowCount/g, this.imgRows.toString() + "u")
                .replace(/colCount/g, this.imgCols.toString() + "u")
                .replace(/filterSize/g, this.filterSize.toString() + "u");

            param.vertexShader = shader_src;
        }
        else {

            param = this.param[param_id];
        }

        if (sub_batch_size == miniBatchSize) {

            param.sub_prev_activation.dt = prev_activation.T().dt;
            param.args["prev_activation"].value = prev_activation.T().dt;
            WebGL2.compute(this.param[param_id]);

            this.z.dt = param.sub_z.T().dt;
            this.activation.dt = param.sub_activation.T().dt;
        }
        else {

            for (var sub_batch_base = 0; sub_batch_base < miniBatchSize; sub_batch_base += sub_batch_size) {

                var all_idx = sub_batch_base;
                var sub_idx = 0;
                for (var i = 0; i < prev_Layer.unitSize; i++) {
                    for (var j = 0; j < sub_batch_size; j++) {
                        param.sub_prev_activation.dt[sub_idx] = prev_activation.dt[all_idx + j];
                        sub_idx++;
                    }
                    all_idx += miniBatchSize;
                }

                param.args["prev_activation"].value = param.sub_prev_activation.dt;
                param.args["z"] = param.sub_z.dt;
                param.args["activation"] = param.sub_activation.dt;
                WebGL2.compute(this.param[param_id]);
                
                all_idx = sub_batch_base;
                sub_idx = 0;
                for (var i = 0; i < this.unitSize; i++) {
                    for (var j = 0; j < sub_batch_size; j++) {
                        this.z.dt[all_idx + j]          = param.sub_z.dt[sub_idx];
                        this.activation.dt[all_idx + j] = param.sub_activation.dt[sub_idx];
                        sub_idx++;
                    }
                    all_idx += miniBatchSize;
                }
            }
        }
    }

    cpuForward() {
        var prev_Layer = this.prevLayer;

        var prev_activation_dt = prev_Layer.activation.dt;
        var z_dt = this.z.dt;
        var activation_dt = this.activation.dt;

        // 出力先
        var output_idx = 0;

        // すべての特徴マップに対し
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            // 出力の行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {

                // 出力の列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {

                    // バッチ内のデータに対し
                    for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                        var sum = 0.0;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {
                                var weight_idx = (feature_idx * this.filterSize + r2) * this.filterSize + c2;
                                var prev_activation_idx = ( (r1 + r2) * prev_Layer.imgCols + (c1 + c2) ) * miniBatchSize + batch_idx;
                                sum += prev_activation_dt[prev_activation_idx] * this.weights.dt[weight_idx];
                            }
                        }

                        var z_val = sum + this.biases.dt[feature_idx];;

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
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {
            for (var r1 = 0; r1 < this.imgRows; r1++) {
                for (var c1 = 0; c1 < this.imgCols; c1++) {
                    for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {
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

            console.log("forward diff:%f cnt:%d GPU:%dms CPU:%dms", max_diff, this.forwardCnt, Math.round(this.forwardGPU / this.forwardCnt), Math.round(this.forwardCPU / this.forwardCnt));
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

        if (this.param[param_id] == undefined) {

            param = {};

            param.id = param_id;

            this.param[param.id] = param;

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

            param = this.param[param_id];
        }

        WebGL2.compute(param);
    }

    cpuNablaWeights(delta_z) {
        var prev_Layer = this.prevLayer;

        // すべての特徴マップに対し
        var delta_z_idx = 0;
        var delta_z_idx_sv = 0;
        var weights_idx = 0;
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            // フィルターの行に対し
            for (var r2 = 0; r2 < this.filterSize; r2++) {

                // フィルターの列に対し
                for (var c2 = 0; c2 < this.filterSize; c2++) {

                    var nabla_w = 0.0;
                    delta_z_idx = delta_z_idx_sv;

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.imgRows; r1++) {

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.imgCols; c1++) {

                            // バッチ内のデータに対し
                            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                                var delta = delta_z.dt[delta_z_idx];
                                if (delta != 0) {

                                    var prev_activation_idx = ((r1 + r2) * prev_Layer.imgCols + (c1 + c2)) * miniBatchSize + batch_idx;
                                    nabla_w += delta * prev_Layer.activation.dt[prev_activation_idx];
                                }
                                delta_z_idx++;
                            }
                        }
                    }

                    this.nablaWeights.dt[weights_idx] = nabla_w;
                    weights_idx++;
                }
            }
            delta_z_idx_sv = delta_z_idx;
        }
        Assert(delta_z_idx == delta_z.dt.length && weights_idx == this.nablaWeights.dt.length);
    }

    cpuNablaBiases(delta_z){
        // すべての特徴マップに対し
        var delta_z_idx = 0;
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            var nabla_b = 0.0;

            // 出力の行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {

                // 出力の列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {

                    // バッチ内のデータに対し
                    for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                        nabla_b += delta_z.dt[delta_z_idx];
                        delta_z_idx++;

                        //!!!!!! 直前が入力層なら必要なし !!!!!
                        // this.costDerivative.dt[output_idx] = this.nextLayer.DeltaT[output_idx];
                    }
                }
            }

            this.nablaBiases.dt[feature_idx] = nabla_b;
        }
        Assert(delta_z_idx == delta_z.dt.length);
    }

    backward(Y, eta2) {
        var lap = new Lap(this.bwTime);

        var delta_z = new ArrayView(this.unitSize, miniBatchSize, new Float32Array(this.nextLayer.deltaX.dt));
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

    updateParameter(eta2) {
        var lap = new Lap(this.udTime);

        var eta3 = eta2;// / (this.filterSize * this.filterSize);

        // すべての特徴マップに対し
        var weights_idx = 0;
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            this.biases.dt[feature_idx] -= eta3 * this.nablaBiases.At2(feature_idx, 0);

            // フィルターの行に対し
            for (var r2 = 0; r2 < this.filterSize; r2++) {

                // フィルターの列に対し
                for (var c2 = 0; c2 < this.filterSize; c2++) {
                    this.weights.dt[weights_idx] -= eta3 * this.nablaWeights.dt[weights_idx];
                    weights_idx++;
                }
            }
        }
        Assert(weights_idx == this.weights.dt.length);
        lap.Time();
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
        this.activation = new ArrayView(this.unitSize, miniBatchSize);
        this.maxIdx     = new Int8Array(this.unitSize * miniBatchSize);
        this.deltaX = new ArrayView(prev_Layer.unitSize, miniBatchSize);
    }

    forward() {
        var lap = new Lap(this.fwTime);

        var prev_Layer = this.prevLayer;

        var prev_activation_dt = prev_Layer.activation.dt;

        // 出力先
        var output_idx = 0;

        // すべての特徴マップに対し
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            // 出力の行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {
                var r0 = r1 * this.filterSize;

                // 出力の列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {
                    var c0 = c1 * this.filterSize;

                    // バッチ内のデータに対し
                    for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                        var max_val = -10000;
                        var max_idx;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {

                                var prev_activation_idx = ( (feature_idx * prev_Layer.imgRows + (r0 + r2)) * prev_Layer.imgCols + (c0 + c2) ) * miniBatchSize + batch_idx;
                                var val = prev_activation_dt[prev_activation_idx];
                                if (max_val < val) {

                                    max_val = val;
                                    max_idx = r2 * this.filterSize + c2;
                                }
                            }
                        }

                        this.activation.dt[output_idx] = max_val;
                        this.maxIdx[output_idx] = max_idx;

                        output_idx++;
                    }
                }
            }
        }
        lap.Time();
    }

    backward(Y, eta2) {
        var lap = new Lap(this.bwTime);

        var prev_Layer = this.prevLayer;

        Assert(this.activation.dt.length == this.nextLayer.deltaX.dt.length);

        for(var i = 0; i < this.deltaX.dt.length; i++){
            this.deltaX.dt[i] = 0;
        }
        lap.Time();

        // 出力先
        var output_idx = 0;

        // すべての特徴マップに対し
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            // 出力の行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {
                var r0 = r1 * this.filterSize;

                // 出力の列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {
                    var c0 = c1 * this.filterSize;

                    // バッチ内のデータに対し
                    for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                        var delta = this.nextLayer.deltaX.dt[output_idx];
                        if(delta != 0){

                            var max_idx = this.maxIdx[output_idx];

                            var r2 = Math.floor(max_idx / this.filterSize);
                            var c2 = max_idx - r2 * this.filterSize;
                            var prev_activation_idx = ( (feature_idx * prev_Layer.imgRows + (r0 + r2)) * prev_Layer.imgCols + (c0 + c2) ) * miniBatchSize + batch_idx;

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

class Network {
    constructor(layers) {
        this.layers = layers;

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

    update_mini_batch(X, Y, eta) {
        this.layers[0].activation = X;
        this.layers.forEach(x => x.forward());

        var eta2 = eta / X.ncol;

        for (var i = this.layers.length - 1; 1 <= i; i--) {
            this.layers[i].backward(Y, eta2);
        }


        this.layers.forEach(x => x.updateParameter(eta2));
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

    evaluate(test_data) {
        var cnt = test_data["count"];
        var labels = test_data["label"];

        var X = new ArrayView(cnt, 28 * 28, TestData["image"]).T();
        this.layers[0].activation = X;
        this.layers.forEach(x => x.forward());

        var result = this.layers[this.layers.length - 1].activation;

        return xrange(cnt).map(c => np.argmax(result.Col(c)) == labels[c] ? 1 : 0).reduce((x, y) => x + y);

//        var test_results = test_data.map($ => { var x = $[0]; var y = $[1]; return [np.argmax(this.feedforward(x)), y]; });
//        return sum(test_results.map($ => {var x = $[0];var y = $[1];return /*int*/(x == y ? 1 : 0);}));
    }

    * SGD(training_data, epochs, mini_batch_size, eta, test_data) {
        miniBatchSize = mini_batch_size;

        for (let j of xrange(epochs)) {

            var idx_list = np.random.RandomSampling(training_data.X.shape[0]);

            var show_time = new Date();
            this.layers.forEach(x => x.miniBatchSizeChanged());

            var training_data_cnt = training_data.X.shape[0];
            var mini_batch_cnt = training_data_cnt / mini_batch_size;
            for (var idx = 0; idx < mini_batch_cnt; idx++) {
                var X = this.Laminate(training_data.X, idx_list, idx * mini_batch_size, mini_batch_size);
                var Y = this.Laminate(training_data.Y, idx_list, idx * mini_batch_size, mini_batch_size);
                this.update_mini_batch(X, Y, eta);
                if (60 * 1000 < new Date() - show_time) {

                    var s = "" + idx + " ";
                    for(let layer of this.layers.slice(1)) {
                        s += " (" + Stats(layer.fwTime, idx) + " " + Stats(layer.bwTime, idx) + " " + Stats(layer.udTime, idx) + ")";
                    }
                    console.log("update mini batch:" + s);
                    yield 1;

                    show_time = new Date();
                }
            }
            yield 2;

            this.layers.forEach(x => x.miniBatchSizeChanged());
            var e = this.evaluate(test_data);

            console.log("Epoch %d: %d / %d ", j, e, test_data["count"]);

            yield 3;
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
