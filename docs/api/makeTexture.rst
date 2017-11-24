makeTexture
===========

構文
^^^^^^

makeTexture(pkg) 

説明
^^^^^^


テクスチャを作ります。


ソース
^^^^^^

.. code-block:: js

        makeTexture(pkg) {
            // すべてのテクスチャに対し
            for (var i = 0; i < pkg.textures.length; i++) {
                var tex_inf = pkg.textures[i];

                // テクスチャのuniform変数の位置
                tex_inf.locTexture = gl.getUniformLocation(pkg.program, tex_inf.name); chk();

                var dim = tex_inf.samplerType == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

                // テクスチャを作る。
                tex_inf.Texture = gl.createTexture(); chk();

                // 指定した位置のテクスチャをアクティブにする。
                gl.activeTexture(this.TEXTUREs[i]); chk();

                // 作成したテクスチャをバインドする。
                gl.bindTexture(dim, tex_inf.Texture); chk();

                if (tex_inf.value instanceof Image) {
                    // テクスチャが画像の場合

                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); chk();
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST); chk();

                    //        gl.texParameteri(gl.TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_MIRRORED_REPEAT); //GL_REPEAT
                    //        gl.texParameteri(gl.TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_MIRRORED_REPEAT); //GL_REPEAT

                    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); chk();
                    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, tex_inf.value); chk();
                    gl.generateMipmap(gl.TEXTURE_2D); chk();
                }
                else {
                    // テクスチャが画像でない場合

                    gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST); chk();
                    gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST); chk();
                    gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); chk();
                    gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); chk();
                }
            }
        }


