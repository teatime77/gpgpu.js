// import
// import
var isDebug = false;
var isFloat64 = false;// true;// isDebug;
var DebugOut = true;

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
        this.Times = v;
    }

    Time(){
        var prev_last_time = this.lastTime;
        this.lastTime = new Date();

        if(this.Times.length <= this.lapIdx){
            this.Times.push(0);
        }
        this.Times[this.lapIdx] += this.lastTime - prev_last_time;
        this.lapIdx++;
    }
}

class Layer {
    clearStats(){
        this.fwTime = [];
        this.bwTime = [];
        this.udTime = [];
    }

    constructor() {
        this.clearStats();
    }

    init(prev_layer) {
        this.prevLayer = prev_layer;
        if (prev_layer) {
            prev_layer.nextLayer = this;
        }
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

    forward() {
        var lap = new Lap(this.fwTime);

        this.batchLength = this.prevLayer.activation.Cols;
        lap.Time();

        this.z = np.dot(this.weight, this.prevLayer.activation).AddV(this.bias);
        lap.Time();

        this.activation = sigmoid(this.z);
        lap.Time();
    }

    backward(Y, eta2) {
        var lap = new Lap(this.bwTime);

        if (!this.nextLayer) {
            // 最後のレイヤーの場合

            this.costDerivative = cost_derivative(this.activation, Y);

            if (isDebug) {

                // cost = 1/2 * Σ xi*xi
                this.cost = xrange(this.costDerivative.Cols).map(c => this.costDerivative.Col(c).dt.map(x => x * x).reduce((x, y) => x + y)).map(x => x / 2);
            }
        }
        else {
            // 最後のレイヤーでない場合

            this.costDerivative = this.nextLayer.deltaX;
        }

        this.Delta = this.costDerivative.Mul(sigmoid_prime(this.z));
        lap.Time();

        this.nabla_b = this.Delta.reduce((x, y) => x + y);
        lap.Time();

        this.nabla_w = np.dot(this.Delta, this.prevLayer.activation.transpose());
        lap.Time();

        if (isDebug) {

            this.nablaBiases = this.Delta;
            // constructor(rows, cols, init, column_major, depth)
            this.nablaWeights = new Mat(this.batchLength, this.weight.Rows, this.weight.Cols);
            for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {
                for (var r = 0; r < this.weight.Rows; r++) {
                    for (var c = 0; c < this.weight.Cols; c++) {
                        var f = this.Delta.At(r, batch_idx) * this.prevLayer.activation.At(c, batch_idx);
                        this.nablaWeights.Set3(batch_idx, r, c, f);
                    }
                }
            }
        }

        //!!!!! 直前が入力層なら必要なし !!!!!
        this.deltaX = np.dot(this.weight.transpose(), this.Delta);
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
        this.biases = new Mat(this.featureCount);
        for (var i = 0; i < this.biases.dt.length; i++) {
            this.biases.dt[i] = np.random.randn();
        }

        this.weights = new Mat(this.featureCount, this.filterSize, this.filterSize);
        for (var i = 0; i < this.weights.dt.length; i++) {
            this.weights.dt[i] = np.random.randn();
        }
    }

    gpuForward() {
        var prev_Layer = this.prevLayer;

        var sub_batch_size;

        if (this.batchLength == 12) {

            sub_batch_size = this.batchLength;
        }
        else if (this.batchLength == 10000) {
            sub_batch_size = 200;// 16;
        }
        else {

            Assert(false);
        }

        var prev_activation = new Mat(prev_Layer.unitSize, this.batchLength, prev_Layer.activation.dt);

        var batch_vec4_count = sub_batch_size / 4;
        var vs_id = "ConvolutionalLayer-forward";
        var param_key = vs_id + ":" + this.filterSize + ":" + this.featureCount + ":" + this.imgRows + ":" + this.imgCols + ":" + sub_batch_size;
        var param;

        if (this.param[param_key] == undefined) {

            param = {};

            param.key = param_key;

            this.param[param.key] = param;

            param.sub_prev_activation = new Mat(prev_Layer.imgRows, prev_Layer.imgCols, sub_batch_size);
            param.sub_z = new Mat(this.unitSize, sub_batch_size);
            param.sub_activation = new Mat(this.unitSize, sub_batch_size);

            param.elementDim   = 4;
            param.elementCount = this.featureCount * this.imgRows * this.imgCols * batch_vec4_count;
            param.textures = [
                { name: "prev_activation", value: param.sub_prev_activation, dim: WebGL2.GL.TEXTURE_3D }
            ];

            param.uniforms = [
                { name: "weights", value: this.weights.dt },
                { name: "biases", value: this.biases.dt }
            ];

            var shader_src = Shaders[vs_id]
                .replace(/featureCount/g, this.featureCount.toString() + "u")
                .replace(/rowCount/g, this.imgRows.toString() + "u")
                .replace(/colCount/g, this.imgCols.toString() + "u")
                .replace(/batchVec4Count/g, batch_vec4_count.toString() + "u")
                .replace(/filterSize/g, this.filterSize.toString() + "u");

            console.log(shader_src);
            param.vsrc = shader_src;

            param.varyings = ["z", "activation"];
            param.arrayBuffers = [param.sub_z, param.sub_activation ];
        }
        else {

            param = this.param[param_key];
        }

        if (sub_batch_size == this.batchLength) {

            param.sub_prev_activation.dt = prev_activation.dt;
            param.sub_z.dt = this.z.dt;
            param.sub_activation.dt = this.activation.dt;
            var ret = WebGL2.compute(this.param[param_key]);
        }
        else {

            for (var sub_batch_base = 0; sub_batch_base < this.batchLength; sub_batch_base += sub_batch_size) {

                var all_idx = sub_batch_base;
                var sub_idx = 0;
                for (var i = 0; i < prev_Layer.unitSize; i++) {
                    for (var j = 0; j < sub_batch_size; j++) {
                        param.sub_prev_activation.dt[sub_idx] = prev_activation.dt[all_idx + j];
                        sub_idx++;
                    }
                    all_idx += this.batchLength;
                }

                var ret = WebGL2.compute(this.param[param_key]);
                
                all_idx = sub_batch_base;
                sub_idx = 0;
                for (var i = 0; i < this.unitSize; i++) {
                    for (var j = 0; j < sub_batch_size; j++) {
                        this.z.dt[all_idx + j]          = param.sub_z.dt[sub_idx];
                        this.activation.dt[all_idx + j] = param.sub_activation.dt[sub_idx];
                        sub_idx++;
                    }
                    all_idx += this.batchLength;
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
                    for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

                        var sum = 0.0;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {
                                var weight_idx = (feature_idx * this.filterSize + r2) * this.filterSize + c2;
                                var prev_activation_idx = ( (r1 + r2) * prev_Layer.imgCols + (c1 + c2) ) * this.batchLength + batch_idx;
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

        if (this.forwardCnt == undefined || this.batchLength != this.prevLayer.activation.Cols) {

            this.forwardCnt = 0;
            this.forwardGPU = 0;
            this.forwardCPU = 0;
        }

        this.batchLength = this.prevLayer.activation.Cols;

        if (!this.z || this.z.Rows != this.unitSize || this.z.Cols != this.batchLength){

            this.z = new Mat(this.unitSize, this.batchLength);
            this.activation = new Mat(this.unitSize, this.batchLength);
        }

        //!!!!!!!!!! テスト用 !!!!!!!!!!
        /*
        if (this.II == undefined) {
            this.II = 0;
        }
        else {
            this.II += 0.1;
        }
        var prev_idx = 0;
        for (var r = 0; r < this.prevLayer.imgRows; r++) {
            for (var c = 0; c < this.prevLayer.imgCols; c++) {
                for (var b = 0; b < this.batchLength; b++) {
                    this.prevLayer.activation.dt[prev_idx] = r * 0.1 + (c + b) * 0.001;
                    prev_idx++;
                }
            }
        }
        // すべての特徴マップに対し
        var weight_idx = 0;
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            this.biases.dt[feature_idx] = 0;//(1 + feature_idx) * 0.1;

            for (var r = 0; r < this.filterSize; r++) {
                for (var c = 0; c < this.filterSize; c++) {
                    this.weights.dt[weight_idx] = (r == 0 && c == 0 ? 1.0 : 0);// + (weight_idx % 10) * 0.1;
                    weight_idx++;
                }
            }
        }
        Assert(weight_idx == this.weights.dt.length);
        */
        //!!!!!!!!!! テスト用 !!!!!!!!!!

        var t0 = new Date();
        this.gpuForward();
        var t1 = new Date();

        lap.Time();

        if(0.1 < Math.random()){

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
                    for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {
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
        if (this.forwardCnt % 100 == 0 || 100 <= this.batchLength) {

            console.log("forward diff:%f cnt:%d GPU:%dms CPU:%dms", max_diff, this.forwardCnt, Math.round(this.forwardGPU / this.forwardCnt), Math.round(this.forwardCPU / this.forwardCnt));
        }
    }

    gpuNablaWeights(delta_z) {
        var prev_Layer = this.prevLayer;

        var prev_activation = new Mat(prev_Layer.imgRows, prev_Layer.imgCols, this.batchLength, prev_Layer.activation.dt);
        var delta_z_3D = new Mat(this.featureCount, this.imgRows, this.imgCols * this.batchLength, delta_z.dt);

        var batch_vec4_count = this.batchLength / 4;
        var vs_id = "ConvolutionalLayer-backward";
        var param_key = vs_id + ":" + this.featureCount + ":" + this.filterSize + ":" + this.imgRows + ":" + this.imgCols + ":" + this.batchLength;
        var param;

        if (this.param[param_key] == undefined) {

            param = {};

            param.key = param_key;

            this.param[param.key] = param;

            param.elementDim   = 1;
            param.elementCount = this.featureCount * this.filterSize * this.filterSize;
            param.textures = [
                { name: "prev_activation", value: prev_activation, dim: WebGL2.GL.TEXTURE_3D },
                { name: "delta_z", value: delta_z_3D, dim: WebGL2.GL.TEXTURE_3D }
            ];

            param.uniforms = [
            ];

            var shader_src = Shaders[vs_id]
                .replace(/featureCount/g, this.featureCount.toString() + "u")
                .replace(/rowCount/g, this.imgRows.toString() + "u")
                .replace(/colCount/g, this.imgCols.toString() + "u")
                .replace(/batchVec4Count/g, batch_vec4_count.toString() + "u")
                .replace(/filterSize/g, this.filterSize.toString() + "u");

            console.log(shader_src);
            param.vsrc = shader_src;

            param.varyings = ["nablaWeights"];
            param.arrayBuffers = [this.nablaWeights ];
        }
        else {

            param = this.param[param_key];
        }

        var ret = WebGL2.compute(param);
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
                            for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

                                var delta = delta_z.dt[delta_z_idx];
                                if (delta != 0) {

                                    var prev_activation_idx = ((r1 + r2) * prev_Layer.imgCols + (c1 + c2)) * this.batchLength + batch_idx;
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
                    for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

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

        var delta_z = new Mat(this.unitSize, this.batchLength, new Float32Array(this.nextLayer.deltaX.dt));
        lap.Time();

        for (var i = 0; i < delta_z.dt.length; i++) {
            delta_z.dt[i] *= sigmoid_primeF(this.z.dt[i]);
        }
        lap.Time();

        this.nablaBiases = new Mat(this.featureCount, 1);
        this.nablaWeights = new Mat(this.featureCount, this.filterSize, this.filterSize);
        this.costDerivative = new Mat(this.unitSize, 1);
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

            this.biases.dt[feature_idx] -= eta3 * this.nablaBiases.At(feature_idx, 0);

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

    forward() {
        var lap = new Lap(this.fwTime);

        var prev_Layer = this.prevLayer;

        this.batchLength = prev_Layer.batchLength;

        var prev_activation_dt = prev_Layer.activation.dt;

        if(this.activation == undefined || this.activation.Cols != this.batchLength ){

            this.activation = new Mat(this.unitSize, this.batchLength);
            this.maxIdx     = new Int8Array(this.unitSize * this.batchLength);

            this.deltaX = new Mat(prev_Layer.unitSize, this.batchLength);
        }

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
                    for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

                        var max_val = -10000;
                        var max_idx;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {

                                var prev_activation_idx = ( (feature_idx * prev_Layer.imgRows + (r0 + r2)) * prev_Layer.imgCols + (c0 + c2) ) * this.batchLength + batch_idx;
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
                    for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

                        var delta = this.nextLayer.deltaX.dt[output_idx];
                        if(delta != 0){

                            var max_idx = this.maxIdx[output_idx];

                            var r2 = Math.floor(max_idx / this.filterSize);
                            var c2 = max_idx - r2 * this.filterSize;
                            var prev_activation_idx = ( (feature_idx * prev_Layer.imgRows + (r0 + r2)) * prev_Layer.imgCols + (c0 + c2) ) * this.batchLength + batch_idx;

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

    gpuBatchTest() {
        var param = {};

        param.elementDim = 4;
        param.elementCount = 16;

        var y = new Float32Array(param.elementDim * param.elementCount);
        var z = new Float32Array(param.elementDim * param.elementCount);

        param.textures = [];
        param.uniforms = [];
        param.vsrc = Shaders["BatchTest"];
        param.varyings = ["y", "z"];
        param.arrayBuffers = [y, z];
        param.key = "BatchTest";

        var ret = WebGL2.compute(param);
    }

    gpuTest() {
        var dt = new Float32Array(4 * 3 * 28 * 28);
        for (var i = 0; i < dt.length; i++) {
            dt[i] = Math.random();// i + 0.123;
        }
        // (rows, cols, init, column_major, depth)
        var m = new Mat(28, 28, 12, dt);
        var param = {};

        param.elementDim = 1;
        param.elementCount = m.dt.length;

        var vs_id = "Test";
        param.textures = [
            { name: "prev_activation", value: m, dim: WebGL2.GL.TEXTURE_3D }
        ];

        var biases = new Float32Array(4);
        for (var i = 0; i < biases.length; i++) {
            biases[i] = i;
        }
        param.uniforms = [
            { name: "biases", value: biases }
        ];

        param.vsrc = Shaders[vs_id];
        param.varyings = ["z", "activation"];
        param.key = vs_id;

        var ret = WebGL2.compute(param);
        for (var i = 0; i < dt.length; i++) {
            Assert(ret[0][i] == i && Math.abs(dt[i] + biases[i % 4] - ret[1][i]) < 0.00001);
        }
        console.log("gpu Test OK");
    }

    Laminate(mini_batch, i) {
        var x_rows = mini_batch[0][i].Rows;
        var X = new Mat(x_rows, mini_batch.length);
        for (var idx = 0; idx < mini_batch.length; idx++) {
            var x = mini_batch[idx][i];
            for (var r = 0; r < x_rows; r++) {
                X.dt[r * X.Cols + idx] = x.dt[r];
            }
        }

        return X;
    }

    check(layer, last_layer, cost_sv, Y, eta2, batch_idx, feature_idx, r, c, max_err) {
        var delta;

        var nabla;
        var param_sv;
        var a_idx;
        var err1, err2, err3;

        if (layer instanceof FullyConnectedLayer) {
            delta = 0.001;

            a_idx = r;
            if (c == -1) {

                nabla = layer.nablaBiases.dt[r];

                param_sv = layer.bias.dt[r];
                layer.bias.dt[r] -= delta;
            }
            else {

                nabla = layer.nablaWeights.At3(batch_idx, r, c);

                param_sv = layer.weight.At(r, c);
                layer.weight.Set(r, c, param_sv - delta);
            }
        }
        else {
            delta = 0.0001;

            if (c == -1) {

                nabla = layer.nablaBiases.At(feature_idx, 0);

                param_sv = layer.biases[feature_idx];
                layer.biases[feature_idx] -= delta;
            }
            else {

                nabla = layer.nablaWeights.At3(feature_idx, r, c);

                param_sv = layer.weights[feature_idx].At(r, c);
                layer.weights[feature_idx].Set(r, c, param_sv - delta);
            }
        }

        this.layers.forEach(x => x.forward());

        for (var i = this.layers.length - 1; 1 <= i; i--) {
            this.layers[i].backward(Y, eta2);
        }

                            
        //-------------------- ΔC
        var deltaC = last_layer.cost[batch_idx] - cost_sv[batch_idx];

        //-------------------- nabla * delta
        var deltaC1 = - nabla * delta;
        err1 = Math.abs((deltaC -deltaC1) / (deltaC == 0 ? 1: deltaC));

        var deltaC2, deltaC3, delta_a, delta_z;
        var cost_deriv;
        var sigmoid_prime_z;

        if (layer instanceof FullyConnectedLayer) {

            //Assert(layer.activation_sv[a_idx] - Y.dt[a_idx] == cost_deriv, "");

            //-------------------- δC/δa0
            delta_a = layer.activation.dt[a_idx] - layer.activation_sv[a_idx];

            cost_deriv = layer.costDerivative_sv[a_idx];
            deltaC2 = delta_a * cost_deriv;

            err2 = Math.abs((deltaC - deltaC2) / (deltaC == 0 ? 1 : deltaC));

            //-------------------- δC/δz0 = δC/δa0 * da0/dz0
            delta_z = layer.z.dt[a_idx] - layer.z_sv[a_idx];

            sigmoid_prime_z = sigmoid_primeF(layer.z_sv[a_idx]);
            deltaC3 = delta_z * cost_deriv * sigmoid_prime_z;

            err3 = Math.abs((deltaC - deltaC3) / (deltaC == 0 ? 1 : deltaC));

            for (var r2 = 0; r2 < layer.nablaBiases.Rows; r2++) {
                if (r2 != r) {
                    Assert(layer.z_sv[r2] - layer.z.dt[r2] == 0 && layer.activation_sv[r2] - layer.activation.dt[r2] == 0, "z-activation-diff");
                }
            }
        }
        else {
            deltaC2 = xrange(layer.activation.dt.length).map(a_idx => (layer.activation.dt[a_idx] - layer.activation_sv[a_idx]) * layer.costDerivative_sv[a_idx]).reduce((x, y) => x + y);

            err2 = Math.abs((deltaC - deltaC2) / (deltaC == 0 ? 1 : deltaC));

            deltaC3 = xrange(layer.activation.dt.length).map(a_idx => (layer.z.dt[a_idx] - layer.z_sv[a_idx]) * layer.costDerivative_sv[a_idx] * sigmoid_primeF(layer.z_sv[a_idx])).reduce((x, y) => x + y);

            err3 = Math.abs((deltaC -deltaC3) / (deltaC == 0 ? 1: deltaC));
        }

        var max_err123 = Math.max(err1, Math.max(err2, err3));
        max_err = Math.max(max_err, max_err123);

        this.ErrSum += max_err123;
        this.ErrCnt++;

        if (10 < max_err123 && DebugOut) {

            console.log("C = 1/2 * Σ(ai - yi)^2 = " + cost_sv[batch_idx]);
            console.log("ΔC = " + deltaC);
            console.log("- nabla * delta = - " + nabla + " * " + delta + " = " + deltaC1);//, layer.prevLayer.activation.At(c, 0)

            if (layer instanceof FullyConnectedLayer) {

                console.log("ΔC ≒ Δa0 * δC/δa0 = " + delta_a + " * " + cost_deriv + " = " + deltaC2);
                console.log("ΔC ≒ Δz0 * δC/δa0 * da0/dz0 = " + delta_z + " * " + cost_deriv + " * " + sigmoid_prime_z + " = " + deltaC3);
            }
            else {

                console.log("ΔC ≒ Δa0 * δC/δa0 = " + deltaC2);
                console.log("ΔC ≒ Δz0 * δC/δa0 * da0/dz0 = " + deltaC3);
            }
            console.log("ΔC誤差 max:" + max_err + " avg:" + (this.ErrSum / this.ErrCnt) + " " + err1 + " " + err2 + " " + err3 + " " + max_err123);
        }

        if (layer instanceof FullyConnectedLayer) {

            if (c == -1) {

                layer.bias.dt[r] = param_sv;
            }
            else {

                layer.weight.Set(r, c, param_sv);
            }
        }
        else {

            if (c == -1) {

                layer.biases[feature_idx] = param_sv;
            }
            else {

                layer.weights[feature_idx].Set(r, c, param_sv);
            }
        }

        return max_err;
    }


    check2(layer, last_layer, cost_sv, Y, eta2, max_err) {      
        var err3 = 0, err_sum = 0, err_cnt = 0;

        //if(layer.nextLayer.maxIdx){

        //    // バッチ内のデータに対し
        //    for (var batch_idx = 0; batch_idx < layer.batchLength; batch_idx++) {
        //        // 出力の行に対し
        //        for (var r1 = 0; r1 < layer.imgRows; r1++) {
        //            // 出力の列に対し
        //            for (var c1 = 0; c1 < layer.imgCols; c1++) {
        //                var output_base = batch_idx * layer.unitSize + layer.featureCount * (r1 * layer.imgCols + c1);

        //                // すべての特徴マップに対し
        //                for (var feature_idx = 0; feature_idx < layer.featureCount; feature_idx++) {

        //                    // 出力先
        //                    var output_idx = output_base + feature_idx;


        //                    this.DeltaT[prev_activation_idx] = deltaT.dt[output_idx];


        //                    var z_val = sum + bias;

        //                    z_dt[output_idx] = z_val;
        //                    activation_dt[output_idx] = sigmoidF(z_val);
        //                }
        //            }
        //        }
        //    }
        //} 

        var batch_idx = 0;
        for (var a_idx = 0; a_idx < layer.activation.dt.length; a_idx++) {
            var sigmoid_prime_z;
            var cost_deriv = layer.costDerivative_sv[a_idx];
            if (cost_deriv != 0) {

                var delta_z;
                var delta_a;

                if (layer.z) {

                    layer.z.dt[a_idx] = layer.z_sv[a_idx];
                    delta_z = layer.z.dt[a_idx] * 0.001;
                    layer.z.dt[a_idx] += delta_z;
                    layer.activation.dt[a_idx] = sigmoidF(layer.z.dt[a_idx]);
                    delta_a = layer.activation.dt[a_idx] - layer.activation_sv[a_idx];
                }
                else{

                    layer.activation.dt[a_idx] = layer.activation_sv[a_idx];
                    delta_a = layer.activation.dt[a_idx] * 0.001;
                    layer.activation.dt[a_idx] += delta_a;
                }

                for (var l = layer.nextLayer; l; l = l.nextLayer) {
                    l.forward();
                }

                if(layer.nextLayer.maxIdx){

                    var next_filter_stride = layer.nextLayer.filterSize * layer.nextLayer.filterSize;
                    var next_a_idx = Math.floor(a_idx / next_filter_stride);
                    if (layer.nextLayer.maxIdx[next_a_idx] != layer.nextLayer.maxIdx_sv[next_a_idx]) {

                        //-------------------- 変更した値を戻す
                        if (layer.z) {

                            layer.z.dt[a_idx] = layer.z_sv[a_idx];
                        }
                        layer.activation.dt[a_idx] = layer.activation_sv[a_idx];

                        continue;
                    }
                }

                for (var i = this.layers.length - 1; 1 <= i; i--) {
                    this.layers[i].backward(Y, eta2);
                }

                //-------------------- ΔC
                var deltaC = last_layer.cost[batch_idx] - cost_sv[batch_idx];

                //-------------------- δC/δa0

                var deltaC2 = delta_a * cost_deriv;

                var err2 = Math.abs((deltaC - deltaC2) / (deltaC == 0 ? 1 : deltaC));

                if (layer.z) {

                    //-------------------- δC/δz0 = δC/δa0 * da0/dz0
                    //var delta_z = layer.z.dt[a_idx] - layer.z_sv[a_idx];
                    sigmoid_prime_z = sigmoid_primeF(layer.z_sv[a_idx]);
                    var deltaC3 = delta_z * cost_deriv * sigmoid_prime_z;

                    err3 = Math.abs((deltaC - deltaC3) / (deltaC == 0 ? 1 : deltaC));
                }

                //-------------------- 誤差表示
                var err23 = Math.max(err2, err3);
                max_err = Math.max(max_err, err23);

                err_sum += err23;
                err_cnt++;
                if (10 < err23 && DebugOut) {

                    console.log("C = " + cost_sv[batch_idx]);
                    console.log("ΔC = " + deltaC);
                    console.log("ΔC ≒ Δa0 * δC/δa0 = " + delta_a + " * " + cost_deriv + " = " + deltaC2);
                    if (layer.z) {

                        console.log("ΔC ≒ Δz0 * δC/δa0 * da0/dz0 = " + delta_z + " * " + cost_deriv + " * " + sigmoid_prime_z + " = " + +deltaC3);
                    }
                    console.log("ΔC誤差  max:" + max_err + " avg:" + (err_sum/err_cnt) + " " + err2 + " " + err3 + " " + err23);
                }

                //-------------------- 変更した値を戻す
                if (layer.z) {

                    layer.z.dt[a_idx] = layer.z_sv[a_idx];
                }
                layer.activation.dt[a_idx] = layer.activation_sv[a_idx];
            }
        }

        return max_err;
    }

    update_mini_batch(X, Y, eta) {
        this.layers[0].activation = X;
        this.layers.forEach(x => x.forward());

        var eta2 = eta / X.Cols;

        for (var i = this.layers.length - 1; 1 <= i; i--) {
            this.layers[i].backward(Y, eta2);
        }


        if (!isDebug) {
            this.layers.forEach(x => x.updateParameter(eta2));
        }
        else {

            var last_layer = this.layers[this.layers.length - 1];
            var cost_sv = new Float32Array( last_layer.cost );
            var max_err = 0;

            this.layers.forEach(layer => {
                if (!(layer instanceof InputLayer)) {
                    if (layer.z) {

                        layer.z_sv = new Float32Array(layer.z.dt);
                    }
                    if (layer.maxIdx) {
                        layer.maxIdx_sv = new Int8Array(layer.maxIdx);
                    }
                    layer.activation_sv = new Float32Array(layer.activation.dt);
                    layer.costDerivative_sv = new Float32Array(layer.costDerivative.dt);
                }
            });

            this.ErrSum = 0;
            this.ErrCnt = 0;

            //for (var layer_idx = this.layers.length - 1; 0 <= layer_idx; layer_idx--) {
            for(var layer_idx = 0; layer_idx < this.layers.length; layer_idx++) {
                var layer = this.layers[layer_idx];
                if (!(layer instanceof InputLayer)) {
                    for (var batch_idx = 0; batch_idx < this.miniBatchSize; batch_idx++) {
                        if (layer instanceof FullyConnectedLayer) {

                            for (var r = 0; r < layer.nablaBiases.Rows; r++) {
                                max_err = this.check(layer, last_layer, cost_sv, Y, eta2, batch_idx, -1, r, -1, max_err);

                                for (var c = 0; c < layer.weight.Cols; c++) {
                                    max_err = this.check(layer, last_layer, cost_sv, Y, eta2, batch_idx, -1, r, c, max_err);
                                }
                            }
                        }
                        else if (layer instanceof PoolingLayer) {

                            max_err = this.check2(layer, last_layer, cost_sv, Y, eta2, max_err);
                        }
                        else {
                            max_err = this.check2(layer, last_layer, cost_sv, Y, eta2, max_err);

                            // すべての特徴マップに対し
                            for (var feature_idx = 0; feature_idx < layer.featureCount; feature_idx++) {

                                max_err = this.check(layer, last_layer, cost_sv, Y, eta2, batch_idx, feature_idx, -1, -1, max_err);

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < layer.filterSize; r2++) {

                                    // フィルターの列に対し
                                    for (var c2 = 0; c2 < layer.filterSize; c2++) {
                                        max_err = this.check(layer, last_layer, cost_sv, Y, eta2, batch_idx, feature_idx, r2, c2, max_err);
                                    }
                                }
                            }
                            /*
                            */
                        }
                    }
                }
            }
        }
    }

    costAvg(cost) {
        return xrange(cost.Cols).map(c => cost.Col(c).dt.map(x => Math.abs(x)).reduce((x, y) => x + y) / cost.Rows);
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

        var X = new Mat(cnt, 28 * 28, TestData["image"]).T();
        this.layers[0].activation = X;
        this.layers.forEach(x => x.forward());

        var result = this.layers[this.layers.length - 1].activation;

        return sum(xrange(cnt).map(c => np.argmax(result.Col(c)) == labels[c] ? 1 : 0));

//        var test_results = test_data.map($ => { var x = $[0]; var y = $[1]; return [np.argmax(this.feedforward(x)), y]; });
//        return sum(test_results.map($ => {var x = $[0];var y = $[1];return /*int*/(x == y ? 1 : 0);}));
    }
}


function* SGD(net, training_data, epochs, mini_batch_size, eta, test_data) {
    //        net.gpuBatchTest();
    net.gpuTest();

    net.miniBatchSize = mini_batch_size;
    var n_test;//??
    if(test_data == undefined){ test_data = None;}
    if(test_data){
        n_test = test_data["count"];
    }
    var n=len(training_data);//??
    for (let j of xrange(epochs)) {

        var start_time = new Date();
        np.random.shuffle(training_data);//??
        console.log("shuffle:" + (new Date() - start_time) + "ms");

        start_time = new Date();
        var mini_batches = xrange(0, n, mini_batch_size).map(k => Slice(training_data, [k, k + mini_batch_size]));//??
        console.log("mini_batches:" + (new Date() - start_time) + "ms");

        start_time = new Date();
        show_time = new Date();
        net.layers.forEach(x => x.clearStats());

        for (var idx = 0; idx < mini_batches.length; idx++) {
            mini_batch = mini_batches[idx];
            var X = net.Laminate(mini_batch, 0);
            var Y = net.Laminate(mini_batch, 1);
            net.update_mini_batch(X, Y, eta);
            show_cnt = (net.layers[1] instanceof ConvolutionalLayer ? 100 : 1000);
            if (10000 < new Date() - show_time) {

                var s = "" + idx + " ";
                for(let layer of net.layers.slice(1)) {
                    s += " (" + Stats(layer.fwTime, idx) + " " + Stats(layer.bwTime, idx) + " " + Stats(layer.udTime, idx) + ")";
                }
                console.log("update mini batch:" + s);
                yield 1;

                show_time = new Date();
            }
        }
        yield 2;

        console.log("update_mini_batch:" + (new Date() - start_time) + "ms");

        if(test_data){
            //??                console.log("Epoch {0}: {1} / {2}".format(j, net.evaluate(test_data), n_test));
            start_time = new Date();
            var e = net.evaluate(test_data);
            console.log("evaluate:" + (new Date() - start_time) + "ms");

            console.log("Epoch %d: %d / %d", j, e, n_test);
        }
        else{
            //??                console.log("Epoch {0} complete".format(j));
            console.log("Epoch %d complete", j);
        }

        yield 3;
    }

    yield 0;
}


function cost_derivative(output_activations, y){
    return (output_activations.Sub(y));
}

function sigmoid(z){
//??    return 1.0 / (1.0 + np.exp(-z));
    return z.map(x => sigmoidF(x));
}

function sigmoid_prime(z){
//??    return sigmoid(z) * (1 - sigmoid(z));
    return z.map(x => sigmoid_primeF(x));
}

function sigmoid_primeF(z) {
    var f = sigmoidF(z);
    return f * (1 - f);
}

//??
function sigmoidF(z){
    return 1.0 / (1.0 + Math.exp(-z));
}

//??
Array.prototype.GetAt = function (i) {
    if (0 <= i) {
        return this[i];
    }
    else {
        return this[this.length + i];
    }
}
