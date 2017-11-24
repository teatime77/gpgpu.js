makeShader
==========

構文
^^^^^^

makeShader(type, source) 

説明
^^^^^^


シェーダを作ります。


ソース
^^^^^^

.. code-block:: js

        makeShader(type, source) {
            source = "#version 300 es\nprecision highp float;\nprecision highp int;\n" + source;

            // シェーダを作る。
            var shader = gl.createShader(type); chk();

            // シェーダにソースをセットする。
            gl.shaderSource(shader, source); chk();

            // シェーダをコンパイルする。
            gl.compileShader(shader); chk();

            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                // コンパイル エラーの場合

                alert(gl.getShaderInfoLog(shader));
                return null;
            }

            return shader;
        }


