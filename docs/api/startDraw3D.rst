startDraw3D
===========

構文
^^^^^^

startDraw3D(draw_obj) 

説明
^^^^^^


3D表示を開始します。


ソース
^^^^^^

.. code-block:: js

        startDraw3D(draw_obj) {
            this.drawObj = draw_obj;
            this.drawParam = {
                xRot : 0,
                yRot : 0,
                z    : -5.0
            }

            var lastMouseX = null;
            var lastMouseY = null;

            // mousemoveのイベント リスナーを登録する。
            this.canvas.addEventListener('mousemove', function (event) {
                var newX = event.clientX;
                var newY = event.clientY;

                if (event.buttons != 0 && lastMouseX != null) {

                    this.drawParam.xRot += (newY -lastMouseY) / 300;
                    this.drawParam.yRot += (newX - lastMouseX) / 300;
                }

                lastMouseX = newX
                lastMouseY = newY;
            }.bind(this));

            // touchmoveのイベント リスナーを登録する。
            this.canvas.addEventListener('touchmove', function (event) {
                // タッチによる画面スクロールを止める
                event.preventDefault(); 

                var newX = event.changedTouches[0].clientX;
                var newY = event.changedTouches[0].clientY;

                if (lastMouseX != null) {

                    this.drawParam.xRot += (newY - lastMouseY) / 300;
                    this.drawParam.yRot += (newX - lastMouseX) / 300;
                }

                lastMouseX = newX
                lastMouseY = newY;
            }.bind(this), false);

            // wheelのイベント リスナーを登録する。
            this.canvas.addEventListener("wheel", function (e) {
                this.drawParam.z += 0.02 * e.deltaY;

                // ホイール操作によるスクロールを無効化する
                e.preventDefault();
            }.bind(this));

            // 3D表示をする。
            this.drawScene();
        }


