drawScene
=========

構文
^^^^^^

drawScene() 

説明
^^^^^^


3D表示をします。


ソース
^^^^^^

.. code-block:: js

        drawScene() {
            var param = this.drawObj.onDraw();

            var pMatrix = mat4.create();
            mat4.perspective(45, this.canvas.width / this.canvas.height, 0.1, 100.0, pMatrix);

            var mvMatrix = mat4.create();
            mat4.identity(mvMatrix);

            mat4.translate(mvMatrix, [0.0, 0.0, this.drawParam.z]);

            mat4.rotate(mvMatrix, this.drawParam.xRot, [1, 0, 0]);
            mat4.rotate(mvMatrix, this.drawParam.yRot, [0, 1, 0]);

            var pmvMatrix = mat4.create();
            mat4.multiply(pMatrix, mvMatrix, pmvMatrix);

            var normalMatrix = mat3.create();
            mat4.toInverseMat3(mvMatrix, normalMatrix);
            mat3.transpose(normalMatrix);

            param.args["uPMVMatrix"] = pmvMatrix;
            param.args["uNMatrix"] = normalMatrix;

            this.compute(param);

            // 次の再描画でdrawSceneが呼ばれるようにする。
            window.requestAnimationFrame(this.drawScene.bind(this));
        }


