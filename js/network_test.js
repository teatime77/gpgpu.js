// JavaScript source code

class FullyConnectedLayerTest extends FullyConnectedLayer {
    backward(Y, eta2) {
        this.nablaBiases = this.Delta;
        // constructor(rows, cols, init, column_major, depth)
        this.nablaWeights = new ArrayView(miniBatchSize, this.weight.nrow, this.weight.ncol);
        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {
            for (var r = 0; r < this.weight.nrow; r++) {
                for (var c = 0; c < this.weight.ncol; c++) {
                    var f = this.Delta.At2(r, batch_idx) * this.prevLayer.activation.At2(c, batch_idx);
                    this.nablaWeights.Set3(batch_idx, r, c, f);
                }
            }
        }
    }
}

class ConvolutionalLayerTest extends ConvolutionalLayer {
    forward() {
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
                for (var b = 0; b < miniBatchSize; b++) {
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
    }
}

class NetworkTest extends NeuralNetwork {

    gpuTest() {
        var dt = new Float32Array(4 * 3 * 28 * 28);
        for (var i = 0; i < dt.length; i++) {
            dt[i] = Math.random();// i + 0.123;
        }
        // (rows, cols, init, column_major, depth)
        var m = new ArrayView(28, 28, 12, dt);
        var z = new Float32Array(m.dt.length);
        var activation = new Float32Array(m.dt.length);

        var biases = new Float32Array(4);
        for (var i = 0; i < biases.length; i++) {
            biases[i] = i;
        }

        var param = {};

        param.elementCount = m.dt.length;

        var vs_id = "Test";
        param.args = {
            "idx_f": MakeFloat32Index(m.dt.length),
            "biases": biases,
            "prev_activation": makeTextureInfo(WebGL2, "vec4", m),
            "z": z,
            "activation": activation,
        };

        param.vertexShader = Shaders[vs_id];
        param.id = vs_id;

        WebGL2.compute(param);
        for (var i = 0; i < dt.length; i++) {
            Assert(z[i] == i && Math.abs(dt[i] + biases[i % 4] - activation[i]) < 0.00001);
        }
        console.log("gpu Test OK");
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

                param_sv = layer.weight.At2(r, c);
                layer.weight.Set2(r, c, param_sv - delta);
            }
        }
        else {
            delta = 0.0001;

            if (c == -1) {

                nabla = layer.nablaBiases.At2(feature_idx, 0);

                param_sv = layer.biases[feature_idx];
                layer.biases[feature_idx] -= delta;
            }
            else {

                nabla = layer.nablaWeights.At3(feature_idx, r, c);

                param_sv = layer.weights[feature_idx].At2(r, c);
                layer.weights[feature_idx].Set2(r, c, param_sv - delta);
            }
        }

        this.layers.forEach(x => x.forward());

        for (var i = this.layers.length - 1; 1 <= i; i--) {
            this.layers[i].backward(Y, eta2);
        }


        //-------------------- ΔC
        var deltaC = last_layer.cost[batch_idx] - cost_sv[batch_idx];

        //-------------------- nabla * delta
        var deltaC1 = -nabla * delta;
        err1 = Math.abs((deltaC - deltaC1) / (deltaC == 0 ? 1 : deltaC));

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

            for (var r2 = 0; r2 < layer.nablaBiases.nrow; r2++) {
                if (r2 != r) {
                    Assert(layer.z_sv[r2] - layer.z.dt[r2] == 0 && layer.activation_sv[r2] - layer.activation.dt[r2] == 0, "z-activation-diff");
                }
            }
        }
        else {
            deltaC2 = xrange(layer.activation.dt.length).map(a_idx => (layer.activation.dt[a_idx] - layer.activation_sv[a_idx]) * layer.costDerivative_sv[a_idx]).reduce((x, y) => x + y);

            err2 = Math.abs((deltaC - deltaC2) / (deltaC == 0 ? 1 : deltaC));

            deltaC3 = xrange(layer.activation.dt.length).map(a_idx => (layer.z.dt[a_idx] - layer.z_sv[a_idx]) * layer.costDerivative_sv[a_idx] * sigmoid_primeF(layer.z_sv[a_idx])).reduce((x, y) => x + y);

            err3 = Math.abs((deltaC - deltaC3) / (deltaC == 0 ? 1 : deltaC));
        }

        var max_err123 = Math.max(err1, Math.max(err2, err3));
        max_err = Math.max(max_err, max_err123);

        this.ErrSum += max_err123;
        this.ErrCnt++;

        if (10 < max_err123) {

            console.log("C = 1/2 * Σ(ai - yi)^2 = " + cost_sv[batch_idx]);
            console.log("ΔC = " + deltaC);
            console.log("- nabla * delta = - " + nabla + " * " + delta + " = " + deltaC1);//, layer.prevLayer.activation.At2(c, 0)

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

                layer.weight.Set2(r, c, param_sv);
            }
        }
        else {

            if (c == -1) {

                layer.biases[feature_idx] = param_sv;
            }
            else {

                layer.weights[feature_idx].Set2(r, c, param_sv);
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
                else {

                    layer.activation.dt[a_idx] = layer.activation_sv[a_idx];
                    delta_a = layer.activation.dt[a_idx] * 0.001;
                    layer.activation.dt[a_idx] += delta_a;
                }

                for (var l = layer.nextLayer; l; l = l.nextLayer) {
                    l.forward();
                }

                if (layer.nextLayer.maxIdx) {

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
                if (10 < err23) {

                    console.log("C = " + cost_sv[batch_idx]);
                    console.log("ΔC = " + deltaC);
                    console.log("ΔC ≒ Δa0 * δC/δa0 = " + delta_a + " * " + cost_deriv + " = " + deltaC2);
                    if (layer.z) {

                        console.log("ΔC ≒ Δz0 * δC/δa0 * da0/dz0 = " + delta_z + " * " + cost_deriv + " * " + sigmoid_prime_z + " = " + +deltaC3);
                    }
                    console.log("ΔC誤差  max:" + max_err + " avg:" + (err_sum / err_cnt) + " " + err2 + " " + err3 + " " + err23);
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

    update_mini_batch(X, Y) {
        var last_layer = this.layers[this.layers.length - 1];
        var cost_sv = new Float32Array(last_layer.cost);
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
        for (var layer_idx = 0; layer_idx < this.layers.length; layer_idx++) {
            var layer = this.layers[layer_idx];
            if (!(layer instanceof InputLayer)) {
                for (var batch_idx = 0; batch_idx < this.miniBatchSize; batch_idx++) {
                    if (layer instanceof FullyConnectedLayer) {

                        for (var r = 0; r < layer.nablaBiases.nrow; r++) {
                            max_err = this.check(layer, last_layer, cost_sv, Y, eta2, batch_idx, -1, r, -1, max_err);

                            for (var c = 0; c < layer.weight.ncol; c++) {
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


function AttribTest(n) {
    var w = new Float32Array(4);
    var ff = new Float32Array(4);
    var x = new Float32Array(4 * 4);
    var y = new Float32Array(4 * 4);
    var z = new Float32Array(4 * 4);
    var biases = new Float32Array(4);
    var tt = new ArrayView(10, 10, 12);

    var idx = 0;
    for (var j = 0; j < 4; j++) {
        w[j] = j;
        ff[j] = j * 10;
        biases[j] = j;
        for (var k = 0; k < 4; k++) {
            x[idx] = idx + n;
            y[idx] = idx * 100;

            idx++;
        }
    }

    for (var i = 0; i < tt.dt.length; i++) {
        tt.dt[i] = i;
    }

    var param = {};

    var vs_id = "AttribTest";

    param.args = {
        "w": w,
        "x": x,
        "y": y,
        "biases": biases,
        "f": n,
        "ff": ff,
        "tt": makeTextureInfo(WebGL2, "vec4", tt),
        "z": z
    };

    param.vertexShader = Shaders[vs_id];
    param.id = vs_id;

    WebGL2.compute(param);

    idx = 0;
    for (var j = 0; j < 4; j++) {
        for (var k = 0; k < 4; k++) {
            Assert(z[idx] == x[idx] + y[idx] + ff[k % 4] + biases[j % 4] + n + (120 + 3 * 12 + 2 * 4 + k));

            idx++;
        }
    }
}
