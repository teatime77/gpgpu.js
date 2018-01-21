// JavaScript source code

var DigitCanvas;

class MNIST {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = this.canvas.getContext('2d');
    }

    onLoadFile(file_name, data) {
        var type, cnt, h, w;

        if (file_name.indexOf("-images") != -1) {
            // 画像の場合

            type = this.BytesToInt(data, 0);

            // データ数
            cnt  = this.BytesToInt(data, 4);

            // 画像の高さ
            h    = this.BytesToInt(data, 8);

            // 画像の幅
            w    = this.BytesToInt(data, 12);

            Assert(16 + cnt * h * w == data.length);

            if (file_name.startsWith("train-images")) {
                // トレーニングの画像の場合

                this.trainingDataImage = data.slice(16);
                this.trainingImgCnt    = cnt;

                this.imgH = h;
                this.imgW = w;

                this.imgIdx = 0;
                setTimeout(this.DrawImage.bind(this), 100);
            }
            else {
                // テストの画像の場合

                this.testDataImage = data.slice(16);
                this.testImgCnt = cnt;
            }

            console.log("画像 name:%s len:%d type:%d cnt:%d H:%d W:%d", file_name, data.length, type, cnt, h, w);
        }
        else if (file_name.indexOf("-labels") != -1) {
            // ラベルの場合

            type = this.BytesToInt(data, 0);

            // データ数
            cnt  = this.BytesToInt(data, 4);

            Assert(8 + cnt == data.length);
            console.log("ラベル name:%s len:%d type:%d cnt:%d", file_name, data.length, type, cnt);

            if (file_name.startsWith("train-labels")) {
                // トレーニングのラベルの場合

                this.trainingDataLabel = data.slice(8);
            }
            else {
                // テストのラベルの場合

                this.testDataLabel = data.slice(8);
            }
        }

        if (this.trainingDataImage && this.trainingDataLabel && this.testDataImage && this.testDataLabel) {

            this.trainingData = this.makeXY(this.trainingImgCnt, this.trainingDataImage, this.trainingDataLabel);
            this.testData     = this.makeXY(this.testImgCnt    , this.testDataImage    , this.testDataLabel);
        }
    }

    makeXY(data_cnt, data_image, data_label) {
        // [0,255] -> [0,1) に変換
        var X = new ArrayView(data_cnt, this.imgH, this.imgW, new Float32Array(data_image).map(a => a / 256.0));

        var Y = new ArrayView(data_cnt, 10);

        // すべてのトレーニングデータに対し
        for (var i = 0; i < data_cnt; i++) {
            // 正解のラベル
            var n = data_label[i];

            // one-hotベクトルの値をセットする。
            Y.dt[i * 10 + n] = 1;
        }

        return { "X": X, "Y": Y };
    }

    BytesToInt(data, offset) {
        return data[offset] * 0x1000000 + data[offset + 1] * 0x10000 + data[offset + 2] * 0x100 + data[offset + 3];
    }

    DrawImage() {
        var image_data = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        var data = image_data.data;
        var length = data.length;

        if (length != 4 * this.imgW * this.imgH) {
            console.log("length(%d) != 4 * ImgW(%d) * ImgH(%d) : %dx%d", length, this.imgW, this.imgH, this.canvas.width, this.canvas.height);
            return;
        }

        var wh = this.imgW * this.imgH;

        for (var i = 0; i < wh; i++) {
            var k = 4 * i;
            var c = 255 - this.trainingDataImage[wh * this.imgIdx + i];

            data[k] = c;
            data[k + 1] = c;
            data[k + 2] = c;
            data[k + 3] = 255;
        }

        this.ctx.putImageData(image_data, 0, 0);

        this.imgIdx += Math.floor( Math.random() * 200 );
        if (this.imgIdx < this.trainingImgCnt) {

            setTimeout(this.DrawImage.bind(this), 1);
        }
    }
}