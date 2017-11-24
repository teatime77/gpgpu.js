compute
=======

構文
^^^^^^

compute(param) 

説明
^^^^^^


計算します。


ソース
^^^^^^

.. code-block:: js

        compute(param) {
            var pkg = this.packages[param.id];
            if (!pkg) {
                // パッケージが未作成の場合

                // パッケージを作る。
                pkg = this.makePackage(param);
            }
            else {

                gl.useProgram(pkg.program); chk();
            }

            // 実引数の値をコピーする。
            this.copyParamArgsValue(param, pkg);

            // attribute変数の値をセットする。
            this.setAttribData(pkg);

            gl.useProgram(pkg.program); chk();

            // テクスチャの値のセットする。
            this.setTextureData(pkg);

            // ユニフォーム変数の値をセットする。
            this.setUniformsData(pkg);

            if (pkg.varyings.length == 0) {
                //  描画する場合

                gl.viewport(0, 0, this.canvas.width, this.canvas.height); chk();

                // カラーバッファと深度バッファをクリアする。
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); chk();

                // 頂点インデックスバッファをバインドする。
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, pkg.VertexIndexBufferInf.buffer); chk();

                // 三角形のリストを描画する。
                gl.drawElements(gl.TRIANGLES, pkg.VertexIndexBufferInf.value.length, gl.UNSIGNED_SHORT, 0); chk();
            }
            else {
                //  描画しない場合

                // ラスタライザを無効にする。
                gl.enable(gl.RASTERIZER_DISCARD); chk();

                // Transform Feedbackをバインドする。
                gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, pkg.transformFeedback); chk();

                // すべてのvarying変数に対し
                for (var i = 0; i < pkg.varyings.length; i++) {
                    var varying = pkg.varyings[i];

                    // Transform Feedbackのバッファをバインドする。
                    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, varying.feedbackBuffer); chk();
                }

                // Transform Feedbackを開始する。
                gl.beginTransformFeedback(gl.POINTS); chk();    // TRIANGLES

                // 点ごとの描画をする。
                gl.drawArrays(gl.POINTS, 0, pkg.attribElementCount); chk();

                // Transform Feedbackを終了する。
                gl.endTransformFeedback(); chk();

                // ラスタライザを有効にする。
                gl.disable(gl.RASTERIZER_DISCARD); chk();

                // すべてのvarying変数に対し
                for (var i = 0; i < pkg.varyings.length; i++) {
                    varying = pkg.varyings[i];

                    // Transform Feedbackのバッファのバインドを解く。
                    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null); chk();

                    // ARRAY_BUFFERにバインドする。
                    gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();

                    // ARRAY_BUFFERのデータを取り出す。
                    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, varying.value); chk();

                    // ARRAY_BUFFERのバインドを解く。
                    gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
                }

                // Transform Feedbackのバインドを解く。
                gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null); chk();
            }

            // プログラムの使用を終了する。
            gl.useProgram(null); chk();
        }


