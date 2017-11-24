makeProgram
===========

構文
^^^^^^

makeProgram(vertex_shader, fragment_shader, varyings) 

説明
^^^^^^


WebGLのプログラムを作ります。


ソース
^^^^^^

.. code-block:: js

        makeProgram(vertex_shader, fragment_shader, varyings) {
            // プログラムを作る。
            var prg = gl.createProgram(); chk();

            // 頂点シェーダをアタッチする。
            gl.attachShader(prg, vertex_shader); chk();

            // フラグメントシェーダをアタッチする。
            gl.attachShader(prg, fragment_shader); chk();

            if (varyings) {
                // varying変数がある場合

                // varying変数の名前の配列
                var varying_names = varyings.map(x => x.name);

                // Transform Feedbackで使うvarying変数を指定する。
                gl.transformFeedbackVaryings(prg, varying_names, gl.SEPARATE_ATTRIBS); chk();   // gl.INTERLEAVED_ATTRIBS 
            }

            // プログラムをリンクする。
            gl.linkProgram(prg); chk();

            if (!gl.getProgramParameter(prg, gl.LINK_STATUS)) {
                // リンクエラーがある場合

                console.log("Link Error:" + gl.getProgramInfoLog(prg));
            }

            // 頂点シェーダを削除する。
            gl.deleteShader(vertex_shader); chk();

            // フラグメントシェーダを削除する。
            gl.deleteShader(fragment_shader); chk();

            return prg;
        }


