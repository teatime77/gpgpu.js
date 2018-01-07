// JavaScript source code
function CreatePlot(canvas_arg) {
    class Plot {
        constructor(canvas) {
            this.canvas = canvas;
            this.context = canvas.getContext('2d');
            this.list = [];
            this.margin = 5;
        }

        clear() {
            this.list = [];
        }

        plot(y, color) {
            this.list.push([y, color]);
        }

        pixX(x) {
            return 30 + (this.canvas.width - (30 + 5)) * (x - this.minX) / this.spanX;
        }

        pixY(y) {
            return 10 + (this.canvas.height - (10 + 5)) * (this.maxY - y) / this.spanY;
        }

        show(aY, aColor) {
            if (aY && aColor) {
                this.clear();
                this.plot(aY, aColor);
            }

            var ctx = this.context;

            ctx.fillStyle = "gainsboro";// "rgb(200, 0, 0)";
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            this.minY = Number.MAX_VALUE;
            this.maxY = -Number.MAX_VALUE;

            this.minX = 0;
            this.maxX = 0;
            for(let p of this.list) {
                var Y, color;
                [Y, color] = p;

                if (Y.length == 0) {
                    continue;
                }
                this.minY = Math.min(this.minY, Y.reduce((u, v) => Math.min(u, v)));
                this.maxY = Math.max(this.maxY, Y.reduce((u, v) => Math.max(u, v)));

                this.maxX = Math.max(this.maxX, Y.length);
            }
            if (this.minY == Number.MAX_VALUE) {
                return;
            }

            var span_y = this.maxY - this.minY;

            if (span_y == 0) {
                return;
            }

            var pr, sc, pr1, pr2, sc1, sc2;

            if (0 < this.maxY) {

                var p = Math.log10(this.maxY);

                // 小数点以下の桁数
                pr1 = Math.floor(p);
                pr2 = pr1 - 1;

                sc1 = Math.pow(10, pr1);
                sc2 = Math.pow(10, pr2);

                var max1 = Math.ceil(this.maxY / sc1) * sc1;
                var max2 = Math.ceil(this.maxY / sc2) * sc2;

                if (0.9 < this.maxY / max1) {

                    pr = pr1;
                    sc = sc1;
                    this.maxY = max1;
                }
                else {

                    pr = pr2;
                    sc = sc2;
                    this.maxY = max2;
                }
            }

            this.spanX = this.maxX - this.minX;
            this.spanY = this.maxY - this.minY;

            for(let p of this.list) {
                var Y, color;
                [Y, color] = p;

                ctx.strokeStyle = color;
                ctx.fillStyle = color;

                ctx.beginPath();
                ctx.moveTo(this.pixX(0), this.pixY(Y[0]));
                for (var i = 1; i < Y.length; i++) {
                    ctx.lineTo(this.pixX(i), this.pixY(Y[i]));
                }
                ctx.stroke();

                for (var i = 0; i < Y.length; i++) {
                    ctx.beginPath();
                    ctx.arc(this.pixX(i), this.pixY(Y[i]), 2, 0, Math.PI * 2, false);
                    ctx.fill();
                }
            }

            ctx.strokeStyle = "black";

            // X軸
            this.drawLine(ctx, this.minX, 0, this.maxX, 0);

            // Y軸
            this.drawLine(ctx, 0, this.minY, 0, this.maxY);

            ctx.font = "16px 'Times New Roman'";

            // Yの最大値
            var txt = (pr < 0 ? this.maxY.toFixed(-pr) : this.maxY.toFixed(0));

            this.rightText(ctx, 0, this.maxY, txt);

            var org_x = this.pixX(0);
            this.drawLinePix(ctx, org_x - 3, this.pixY(this.maxY), org_x + 3, this.pixY(this.maxY));

            // Yの目盛り
            for (var i = 1; i * sc1 < this.maxY; i++) {

                var y = i * sc1;
                var txt = (pr1 < 0 ? y.toFixed(-pr1) : y.toFixed(0));
                this.rightText(ctx, 0, y, txt);

                this.drawLinePix(ctx, org_x - 3, this.pixY(y), org_x + 3, this.pixY(y));
            }
        }

        drawLinePix(ctx, x1, y1, x2, y2) {
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }

        drawLine(ctx, x1, y1, x2, y2) {
            this.drawLinePix(ctx, this.pixX(x1), this.pixY(y1), this.pixX(x2), this.pixY(y2));
        }

        rightText(ctx, x, y, txt) {
            var tm = ctx.measureText(txt);

            ctx.strokeText(txt, this.pixX(x) - this.margin - tm.width, this.pixY(y) + 8);
        }
    }

    return new Plot(canvas_arg);
}
