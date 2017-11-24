GPGPU.constructor
=================

構文
^^^^^^

constructor(canvas) 

説明
^^^^^^


GPGPUのコンストラクタ


ソース
^^^^^^

.. code-block:: js

        constructor(canvas) {
            console.log("init WebGL");

            if (!canvas) {
                // canvasが指定されていない場合

                // canvasを作る。
                canvas = document.createElement('canvas');
                
                // canvasをサイズをセットする。
                canvas.width = 32;
                canvas.height = 32;

                // canvasをdocumentに追加する。
                document.body.appendChild(canvas);
            }

            this.canvas = canvas;

            // canvasからWebGL2のcontextを得る。
            gl = canvas.getContext('webgl2', { antialias: false });
            var isWebGL2 = !!gl;
            if (!isWebGL2) {
                // WebGL2のcontextを得られない場合

                console.log("WebGL 2 is not available. See How to get a WebGL 2 implementation");
                console.log("https://www.khronos.org/webgl/wiki/Getting_a_WebGL_Implementation");

                throw "WebGL 2 is not available.";
            }

            // パッケージのリストを初期化する。
            this.packages = {};

            // 標準のシェーダの文字列をセットする。
            this.setStandardShaderString();

            this.TEXTUREs = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3];
        }


