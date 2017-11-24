clear
=====

構文
^^^^^^

clear(id) 

説明
^^^^^^


指定したidのWebGLのオブジェクトをすべて削除します。


ソース
^^^^^^

.. code-block:: js

        clear(id) {
            var pkg = this.packages[id];

            if (pkg) {
                // 指定したidのパッケージがある場合

                delete this.packages[id]

                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();

                if (pkg.idxBuffer) {

                    // バッファを削除する。
                    gl.deleteBuffer(pkg.idxBuffer); chk();
                }

                // すべてのvarying変数に対し
                for (let varying of pkg.varyings) {
                    if (varying.feedbackBuffer) {
                        // Transform Feedbackバッファがある場合

                        // バッファを削除する。
                        gl.deleteBuffer(varying.feedbackBuffer); chk();
                    }
                }

                if (pkg.transformFeedback) {
                    // Transform Feedbackがある場合

                    gl.deleteTransformFeedback(pkg.transformFeedback); chk();
                }

                // テクスチャのバインドを解く。
                gl.bindTexture(gl.TEXTURE_2D, null); chk();
                gl.bindTexture(gl.TEXTURE_3D, null); chk();

                // すべてのテクスチャを削除する。
                pkg.textures.forEach(x => gl.deleteTexture(x.Texture), chk())

                // プログラムを削除する。
                gl.deleteProgram(pkg.program); chk();
            }
        }


