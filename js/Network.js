// import
// import

class Layer {
    constructor() {
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
        this.z = np.dot(this.weight, this.prevLayer.activation).AddV(this.bias);
        this.activation = sigmoid(this.z);
    }

    backward(Y, eta2) {
        if (!this.nextLayer) {

            this.Delta = cost_derivative(this.activation, Y).Mul(sigmoid_prime(this.z));
        }
        else {

            this.Delta = np.dot(this.nextLayer.weight.transpose(), this.nextLayer.Delta).Mul(sigmoid_prime(this.z));
        }
        this.nabla_b = this.Delta.reduce((x, y) => x + y);
        this.nabla_w = np.dot(this.Delta, this.prevLayer.activation.transpose());
    }

    updateParameter(eta2) {
        this.weight = this.weight.Sub(eta2.Mul(this.nabla_w));
        this.bias = this.bias.Sub(eta2.Mul(this.nabla_b));
    }
}

class ConvolutionalLayer extends Layer{
    constructor(filter_size, filter_count) {
        super();

        this.filterSize = filter_size;
        this.filterCount = filter_count;
    }

    init(prev_layer) {
        super.init(prev_layer);

        Assert(this.prevLayer instanceof InputLayer, "Convolutional-Layer-init");

        this.imgRows = this.prevLayer.imgRows - this.filterSize + 1;
        this.imgCols = this.prevLayer.imgCols - this.filterSize + 1;
        this.unitSize = this.imgRows * this.imgCols * this.filterCount;

        this.biases = xrange(this.filterCount).map(x => np.random.randn());
        this.weights = xrange(this.filterCount).map(x => np.random.randn(this.filterSize, this.filterSize));
    }

    forward() {
        var prev_Layer = this.prevLayer;
        this.batchLength = prev_Layer.activation.Cols;
        var prev_activation = prev_Layer.activation.dt;
        var in_dt_part = new Float32Array(this.filterSize * this.filterSize);

        this.z = new Mat(this.unitSize, this.batchLength, null, true);
        for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

            for (var r = 0; r < this.imgRows; r++) {
                for (var c = 0; c < this.imgCols; c++) {

                    for (var r2 = 0; r2 < this.filterSize; r2++) {
                        for (var c2 = 0; c2 < this.filterSize; c2++) {
                            var k = batch_idx * prev_Layer.unitSize + (r + r2) * prev_Layer.imgCols + (c + c2);
                            in_dt_part[r2 * this.filterSize + c2] = prev_activation[k];
                        }
                    }

                    var output_base = batch_idx * this.unitSize + this.filterCount * (r * this.imgCols + c);

                    for(var filter_idx = 0; filter_idx < this.filterCount; filter_idx++){

                        var weight = this.weights[filter_idx];
                        var bias = this.biases[filter_idx];

                        var sum = 0.0;
                        for (var r2 = 0; r2 < this.filterSize; r2++) {
                            for (var c2 = 0; c2 < this.filterSize; c2++) {
                                var j = r2 * this.filterSize + c2;
                                sum += in_dt_part[j] * weight.dt[j];
                            }
                        }

                        this.z.dt[output_base + filter_idx] = sum + bias;
                    }
                }
            }
        }

        this.activation = sigmoid(this.z);
    }

    backward(Y, eta2) {
        this.Delta = this.nextLayer.Delta.Mul(sigmoid_prime(this.z));

        var prev_Layer = this.prevLayer;
        var prev_activation = prev_Layer.activation.dt;

        for(var filter_idx = 0; filter_idx < this.filterCount; filter_idx++) {

            var nabla_b = 0.0;
            for (var r2 = 0; r2 < this.filterSize; r2++) {
                for (var c2 = 0; c2 < this.filterSize; c2++) {

                    var nabla_w = 0.0;

                    for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

                        for (var r = 0; r < this.imgRows; r++) {
                            for (var c = 0; c < this.imgCols; c++) {

                                var output_base = batch_idx * this.unitSize + this.filterCount * (r * this.imgCols + c);
                                var out_idx = output_base + filter_idx;

                                var delta = this.Delta.dt[out_idx];
                                if (delta != 0) {

                                    var k = batch_idx * prev_Layer.unitSize +(r + r2) * prev_Layer.imgCols + (c + c2);

                                    nabla_w += delta * prev_activation[k];

                                    nabla_b += delta;
                                }
                            }
                        }
                    }

                    var j = r2 * this.filterSize + c2;
                    this.weights[filter_idx].dt[j] -= eta2 * nabla_w;
                }
            }

            this.biases[filter_idx] -= eta2 * nabla_b;
        }
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
        this.filterCount = this.prevLayer.filterCount;

        this.unitSize = this.imgRows * this.imgCols * this.filterCount;
    }

    forward() {
        var prev_Layer = this.prevLayer;
        var prev_activation_dt = prev_Layer.activation.dt;
        this.batchLength = prev_Layer.batchLength;
        var out_dt = new Float32Array(this.unitSize * this.batchLength);

        this.maxIdx = new Int8Array(this.unitSize * this.batchLength);

        for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

            for (var r = 0; r < this.imgRows; r++) {
                for (var c = 0; c < this.imgCols; c++) {

                    var output_base = batch_idx * this.unitSize + this.filterCount * (r * this.imgCols + c);

                    for (var filter_idx = 0; filter_idx < this.filterCount; filter_idx++) {

                        var max_val = -10000;
                        var max_idx;
                        for (var r2 = 0; r2 < this.filterSize; r2++) {
                            for (var c2 = 0; c2 < this.filterSize; c2++) {

                                var k = batch_idx * prev_Layer.unitSize + prev_Layer.filterCount * ((r + r2) * prev_Layer.imgCols + (c + c2)) + filter_idx;
                                var val = prev_activation_dt[k];
                                if (max_val < val) {

                                    max_val = val;
                                    max_idx = r2 * this.filterSize + c2;
                                }
                            }
                        }

                        out_dt[output_base + filter_idx] = max_val;
                        this.maxIdx[output_base + filter_idx] = max_idx;
                    }
                }
            }
        }

        this.activation = (new Mat(this.batchLength, this.unitSize, out_dt)).T();
    }

    backward(Y, eta2) {
        var filter_stride = this.filterSize * this.filterSize;
        var delta = np.dot(this.nextLayer.weight.transpose(), this.nextLayer.Delta);
        Assert(delta.Rows == this.unitSize && delta.Cols == this.batchLength, "Pooling-Layer-backward");

        var dt = new Float32Array(this.prevLayer.activation.dt.length);

        for (var i = 0; i < delta.dt.length; i++) {
            var k = i * filter_stride + this.maxIdx[i];
            dt[k] = delta.dt[i];
        }

        this.Delta = new Mat(this.prevLayer.unitSize, this.batchLength, dt, true);
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

    SGD(training_data, epochs, mini_batch_size, eta, test_data) {
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
//                console.log("update mini batch:%d", j);//??
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

    update_mini_batch(X, Y, eta) {
        this.layers[0].activation = X;
        this.layers.forEach(x => x.forward());

        var eta2 = eta / X.Cols;

        for (var i = this.layers.length - 1; 1 <= i; i--) {
            this.layers[i].backward(Y, eta2);
        }

        this.layers.forEach(x => x.updateParameter(eta2));
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
    return sigmoidF(z) * (1 - sigmoidF(z));
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
