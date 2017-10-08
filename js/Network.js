// import
// import
var isDebug = false;
var isFloat64 = false;// true;// isDebug;
var DebugOut = true;

function newFloatArray(x) {
    if(isFloat64){
        return new Float64Array(x);
    }
    else {

        return new Float32Array(x);
    }
}

class Layer {
    constructor() {
        this.fwCnt = 0;
        this.fwTime = 0;
        this.bwCnt = 0;
        this.bwTime = 0;
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

    forward2() {
        var startTime = new Date();
        this.forward();
        this.fwCnt++;
        this.fwTime += new Date() - startTime;
    }

    backward2(Y, eta2) {
        var startTime = new Date();
        this.backward(Y, eta2);
        this.bwCnt++;
        this.bwTime += new Date() - startTime;
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
        this.batchLength = this.prevLayer.activation.Cols;
        this.z = np.dot(this.weight, this.prevLayer.activation).AddV(this.bias);
        this.activation = sigmoid(this.z);
    }

    backward(Y, eta2) {
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

        this.nabla_b = this.Delta.reduce((x, y) => x + y);
        this.nabla_w = np.dot(this.Delta, this.prevLayer.activation.transpose());

        if (isDebug) {

            this.nablaBiases = this.Delta;
            // constructor(rows, cols, init, column_major, depth)
            this.nablaWeights = new Mat(this.weight.Rows, this.weight.Cols, null, false, this.batchLength);
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
    }

    updateParameter(eta2) {
        this.weight = this.weight.Sub(eta2.Mul(this.nabla_w));
        this.bias = this.bias.Sub(eta2.Mul(this.nabla_b));
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

        this.weights = new Mat(this.filterSize, this.filterSize, null, false, this.featureCount);
        for (var i = 0; i < this.weights.dt.length; i++) {
            this.weights.dt[i] = np.random.randn();
        }
    }

    gpuForward() {
        var prev_Layer = this.prevLayer;

        var sub_batch_size;

        if (this.batchLength == 12 || true) {//!!!!!!!!!! テスト用 !!!!!!!!!!

            sub_batch_size = this.batchLength;
        }
        else if (this.batchLength == 10000) {
            sub_batch_size = 16;
        }
        else {

            Assert(false);
        }

        var prev_activation = new Mat(prev_Layer.unitSize, this.batchLength, prev_Layer.activation.dt);


        if (sub_batch_size == this.batchLength) {

            this.sub_z = this.z;
            this.sub_activation = this.activation;
        }
        else {
            this.sub_z = new Mat(sub_batch_size, this.unitSize);
            this.sub_activation = new Mat(sub_batch_size, this.unitSize);
        }

        var batch_vec4_count = sub_batch_size / 4;
        var param_key = vs_id + ":" + this.filterSize + ":" + this.featureCount + ":" + this.imgRows + ":" + this.imgCols + ":" + sub_batch_size;
        var param;
        if (this.param[param_key] == undefined) {

            // (rows, cols, init, column_major, depth)


            var vs_id = "ConvolutionalLayer-forward";

            param = {};

            param.key = param_key;

            this.param[param.key] = param;

            param.sub_prev_activation = new Mat(prev_Layer.imgCols, sub_batch_size, undefined, false, prev_Layer.imgRows);

            param.elementDim   = 4;
            param.elementCount = this.featureCount * this.imgRows * this.imgCols * batch_vec4_count;
            param.textures = [
                { name: "prev_activation", value: param.sub_prev_activation, dim: WebGL2.GL.TEXTURE_3D }
            ];

            param.uniforms = [
                { name: "weights", value: this.weights.dt },
                { name: "biases", value: this.biases.dt }
            ];

            if (this.forwardSrc == undefined) {

                this.forwardSrc = Shaders[vs_id]

                    .replace(/featureCount/g, this.featureCount.toString() + "u")
                    .replace(/rowCount/g, this.imgRows.toString() + "u")
                    .replace(/colCount/g, this.imgCols.toString() + "u")
                    .replace(/batchVec4Count/g, batch_vec4_count.toString() + "u")
                    .replace(/filterSize/g, this.filterSize.toString() + "u");
            }
            console.log(this.forwardSrc);
            param.vsrc = this.forwardSrc;

            param.varyings = ["z", "activation"];
            param.arrayBuffers = [this.sub_z.dt, this.sub_activation.dt];
        }
        else {

            param = this.param[param_key];
        }

        for (var sub_batch_base = 0; sub_batch_base < this.batchLength; sub_batch_base += sub_batch_size) {

            //!!!!!!!!!!!!!!!!!!!!!!!!!!
            var src = sub_batch_base;
            var dst = 0;
            for (var i = 0; i < prev_Layer.unitSize; i++) {
                for (var j = 0; j < sub_batch_size; j++) {
                    param.sub_prev_activation.dt[dst] = prev_activation.dt[src + j];
                    dst++;
                }
                src += sub_batch_size;
            }

            var ret = WebGL2.Calc(this.param[param_key]);

            if (sub_batch_size != this.batchLength) {

                this.sub_z.CopyRows(this.z, 0, sub_batch_base, sub_batch_size);
                this.sub_activation.CopyRows(this.activation, 0, sub_batch_base, sub_batch_size);
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
                    this.prevLayer.activation.dt[prev_idx] = Math.random();// + r * 0.1 + (c + b) * 0.001;
                    prev_idx++;
                }
            }
        }
        // すべての特徴マップに対し
        var weight_idx = 0;
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            this.biases.dt[feature_idx] = Math.random();// + feature_idx * 0.1;

            for (var r = 0; r < this.filterSize; r++) {
                for (var c = 0; c < this.filterSize; c++) {
                    this.weights.dt[weight_idx] = Math.random();// 1.23 + (weight_idx % 10) * 0.1;
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

        var z_gpu_dt = newFloatArray(this.z.dt);
        var activation_gpu_dt = newFloatArray(this.activation.dt);

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
                            max_diff = diff;
                        }
                        output_idx++;
                    }
                }
            }
        }
        Assert(max_diff < 0.00001, "Convolutional-Layer-forward-diff");
        if (this.forwardCnt == undefined) {

            this.forwardCnt = 0;
            this.forwardGPU = 0;
            this.forwardCPU = 0;
        }
        this.forwardCnt++;
        this.forwardGPU += t1 - t0;
        this.forwardCPU += t2 - t1;
        if (this.forwardCnt % 100 == 0) {

            console.log("forward diff:%f cnt:%d GPU:%dms CPU:%dms", max_diff, this.forwardCnt, Math.round(this.forwardGPU / this.forwardCnt), Math.round(this.forwardCPU / this.forwardCnt));
        }
    }

    backward(Y, eta2) {
        var batch_length = this.batchLength;
        //this.Delta = this.nextLayer.Delta.Mul(sigmoid_prime(this.z));
        var delta_z = new Mat(this.unitSize, batch_length, new Float32Array(this.nextLayer.deltaX.dt));

        for (var i = 0; i < delta_z.dt.length; i++) {
            delta_z.dt[i] *= sigmoid_primeF(this.z.dt[i]);
        }

        var prev_Layer = this.prevLayer;

        this.nablaBiases = new Mat(this.featureCount, 1);
        this.nablaWeights = new Mat(this.filterSize, this.filterSize, null, false, this.featureCount);
        this.costDerivative = new Mat(this.unitSize, 1);


        // すべての特徴マップに対し
        var delta_z_idx = 0;
        for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

            var nabla_b = 0.0;

            // 出力の行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {

                // 出力の列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {

                    // バッチ内のデータに対し
                    for (var batch_idx = 0; batch_idx < batch_length; batch_idx++) {

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

        // すべての特徴マップに対し
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

    updateParameter(eta2) {
        var eta3 = eta2 / (this.filterSize * this.filterSize);

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

        this.imgRows = this.prevLayer.imgRows / this.filterSize;
        this.imgCols = this.prevLayer.imgCols / this.filterSize;
        this.featureCount = this.prevLayer.featureCount;

        this.unitSize = this.imgRows * this.imgCols * this.featureCount;
    }

    forward() {
        var prev_Layer = this.prevLayer;
        var prev_activation_dt = prev_Layer.activation.dt;
        this.batchLength = prev_Layer.batchLength;
        var out_dt = newFloatArray(this.unitSize * this.batchLength);

        this.maxIdx = new Int8Array(this.unitSize * this.batchLength);

        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

            // すべての行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {

                // すべての列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {

                    var output_base = batch_idx * this.unitSize + this.featureCount * (r1 * this.imgCols + c1);

                    // すべての特徴マップに対し
                    for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

                        var max_val = -10000;
                        var max_idx;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {

                                var prev_activation_idx = batch_idx * prev_Layer.unitSize + prev_Layer.featureCount * ((r1 + r2) * prev_Layer.imgCols + (c1 + c2)) + feature_idx;
                                var val = prev_activation_dt[prev_activation_idx];
                                if (max_val < val) {

                                    max_val = val;
                                    max_idx = r2 * this.filterSize + c2;
                                }
                            }
                        }

                        // 出力先
                        var output_idx = output_base + feature_idx;

                        out_dt[output_idx] = max_val;
                        this.maxIdx[output_idx] = max_idx;
                    }
                }
            }
        }

        this.activationT = new Mat(this.batchLength, this.unitSize, out_dt);
        this.activation = this.activationT.T();
    }

    backward(Y, eta2) {
        var prev_Layer = this.prevLayer;

        this.costDerivative = np.dot(this.nextLayer.weight.transpose(), this.nextLayer.Delta);
        var deltaT = this.costDerivative.T();
        Assert(deltaT.Rows == this.batchLength && deltaT.Cols == this.unitSize, "Pooling-Layer-backward");

        this.DeltaT = newFloatArray(this.prevLayer.activation.dt.length);

        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

            // すべての行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {

                // すべての列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {

                    var output_base = batch_idx * this.unitSize + this.featureCount * (r1 * this.imgCols + c1);

                    // すべての特徴マップに対し
                    for (var feature_idx = 0; feature_idx < this.featureCount; feature_idx++) {

                        // 出力先
                        var output_idx = output_base + feature_idx;
                        var max_idx = this.maxIdx[output_idx];
                        var r2 = Math.floor(max_idx / this.filterSize);
                        var c2 = max_idx - r2 * this.filterSize;
                        var prev_activation_idx = batch_idx * prev_Layer.unitSize + prev_Layer.featureCount * ((r1 + r2) * prev_Layer.imgCols + (c1 + c2)) + feature_idx;

                        this.DeltaT[prev_activation_idx] = deltaT.dt[output_idx];
                    }
                }
            }
        }
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

        var ret = WebGL2.Calc(param);
    }

    gpuTest() {
        var dt = newFloatArray(4 * 3 * 28 * 28);
        for (var i = 0; i < dt.length; i++) {
            dt[i] = Math.random();// i + 0.123;
        }
        // (rows, cols, init, column_major, depth)
        var m = new Mat(28, 12, dt, false, 28);
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

        var ret = WebGL2.Calc(param);
        for (var i = 0; i < dt.length; i++) {
            Assert(ret[0][i] == i && Math.abs(dt[i] + biases[i % 4] - ret[1][i]) < 0.00001);
        }
        console.log("gpu Test OK");
    }

    SGD(training_data, epochs, mini_batch_size, eta, test_data) {
//        this.gpuBatchTest();
        this.gpuTest();

        this.miniBatchSize = mini_batch_size;
        var n_test;//??
        if(test_data == undefined){ test_data = None;}
        if(test_data){
            n_test = test_data["count"];
        }
        var n=len(training_data);//??
        for (let j of xrange(epochs)) {

            var startTime = new Date();
            np.random.shuffle(training_data);//??
            console.log("shuffle:" + (new Date() - startTime) + "ms");

            startTime = new Date();
            var mini_batches = xrange(0, n, mini_batch_size).map(k => Slice(training_data, [k, k + mini_batch_size]));//??
            console.log("mini_batches:" + (new Date() - startTime) + "ms");

            startTime = new Date();
            for (let mini_batch of mini_batches) {
                var X = this.Laminate(mini_batch, 0);
                var Y = this.Laminate(mini_batch, 1);
                this.update_mini_batch(X, Y, eta);
                if (this.layers[1].fwCnt % 1000 == 0) {

                    var s = "" + this.layers[1].fwCnt + " ";
                    for(let layer of this.layers.slice(1)) {
                        s += " (" + Math.floor(layer.fwTime / layer.fwCnt) + " " + Math.floor(layer.bwTime / layer.bwCnt) + ")";
                    }
                    console.log("update mini batch:" + s);

                    if (this.layers[1] instanceof ConvolutionalLayer) {

                        break;//!!!!!!!!!!!!!!!!!!!! テスト用 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    }
                }
            }
            console.log("update_mini_batch:" + (new Date() - startTime) + "ms");

            if(test_data){
                //??                console.log("Epoch {0}: {1} / {2}".format(j, this.evaluate(test_data), n_test));
                startTime = new Date();
                var e = this.evaluate(test_data);
                console.log("evaluate:" + (new Date() - startTime) + "ms");

                console.log("Epoch %d: %d / %d", j, e, n_test);
            }
            else{
//??                console.log("Epoch {0} complete".format(j));
                console.log("Epoch %d complete", j);
            }
        }
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

        this.layers.forEach(x => x.forward2());

        for (var i = this.layers.length - 1; 1 <= i; i--) {
            this.layers[i].backward2(Y, eta2);
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
                    l.forward2();
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
                    this.layers[i].backward2(Y, eta2);
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
        this.layers.forEach(x => x.forward2());

        var eta2 = eta / X.Cols;

        for (var i = this.layers.length - 1; 1 <= i; i--) {
            this.layers[i].backward2(Y, eta2);
        }


        if (!isDebug) {
            this.layers.forEach(x => x.updateParameter(eta2));
        }
        else {

            var last_layer = this.layers[this.layers.length - 1];
            var cost_sv = newFloatArray( last_layer.cost );
            var max_err = 0;

            this.layers.forEach(layer => {
                if (!(layer instanceof InputLayer)) {
                    if (layer.z) {

                        layer.z_sv = newFloatArray(layer.z.dt);
                    }
                    if (layer.maxIdx) {
                        layer.maxIdx_sv = new Int8Array(layer.maxIdx);
                    }
                    layer.activation_sv = newFloatArray(layer.activation.dt);
                    layer.costDerivative_sv = newFloatArray(layer.costDerivative.dt);
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
