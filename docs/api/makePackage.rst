makePackage
===========

構文
^^^^^^

makePackage(param) 

説明
^^^^^^


パッケージを作ります。


ソース
^^^^^^

.. code-block:: js

        makePackage(param) {
            var pkg = {};
            this.packages[param.id] = pkg;

            pkg.id = param.id;

            if (!param.fragmentShader) {
                // フラグメントシェーダが指定されてない場合

                // デフォルトのフラグメントシェーダをセットする。
                param.fragmentShader = this.minFragmentShader;
            }

            // シェーダのソースコードを解析する。
            this.parseShader(pkg, param);

            // 頂点シェーダを作る。
            var vertex_shader = this.makeShader(gl.VERTEX_SHADER, param.vertexShader);

            // フラグメントシェーダを作る。
            var fragment_shader = this.makeShader(gl.FRAGMENT_SHADER, param.fragmentShader);

            // プログラムを作る。
            pkg.program = this.makeProgram(vertex_shader, fragment_shader, pkg.varyings);

            // プログラムを使用する。
            gl.useProgram(pkg.program); chk();

            // ユニフォーム変数のロケーションをセットします。
            this.setUniformLocation(pkg);

            // テクスチャを作る。
            this.makeTexture(pkg);

            pkg.attribElementCount = param.elementCount;

            // attribute変数を作る。
            this.makeAttrib(pkg);

            if (pkg.varyings.length != 0) {
                //  varying変数がある場合

                // すべてのvarying変数に対し
                for (let varying of pkg.varyings) {
                    var out_buffer_size = this.vecDim(varying.type) * pkg.attribElementCount * Float32Array.BYTES_PER_ELEMENT;

                    // Transform Feedbackバッファを作る。
                    varying.feedbackBuffer = gl.createBuffer(); chk();

                    // バッファをバインドする。
                    gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();
                    gl.bufferData(gl.ARRAY_BUFFER, out_buffer_size, gl.STATIC_COPY); chk();
                    gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
                }

                // Transform Feedbackを作る。
                pkg.transformFeedback = gl.createTransformFeedback(); chk();
            }

            if (param.VertexIndexBuffer) {
                this.makeVertexIndexBuffer(pkg, param);
            }

            return pkg;
        }


