setTextureData
==============

構文
^^^^^^

setTextureData(pkg) 

説明
^^^^^^


テクスチャのデータをセットします。


ソース
^^^^^^

.. code-block:: js

        setTextureData(pkg) {
            for (var i = 0; i < pkg.textures.length; i++) {
                var tex_inf = pkg.textures[i];

                // テクスチャのuniform変数にテクスチャの番号をセットする。
                gl.uniform1i(tex_inf.locTexture, i); chk();

                var dim = tex_inf.samplerType == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

                // 指定した位置のテクスチャをアクティブにする。
                gl.activeTexture(this.TEXTUREs[i]); chk();

                // テクスチャをバインドする。
                gl.bindTexture(dim, tex_inf.Texture); chk();

                if (tex_inf.value instanceof Image) {
                    // テクスチャが画像の場合

                }
                else {
                    // テクスチャが画像でない場合

                    var internal_format, format;
                    switch (tex_inf.texelType) {
                        case "float":
                            internal_format = gl.R32F;
                            format = gl.RED;
                            break;

                        case "vec2":
                            internal_format = gl.RG32F;
                            format = gl.RG;
                            break;

                        case "vec3":
                            internal_format = gl.RGB32F;
                            format = gl.RGB;
                            break;

                        case "vec4":
                            internal_format = gl.RGBA32F;
                            format = gl.RGBA;
                            break;

                        default:
                            assert(false);
                            break;
                    }

                    if (dim == gl.TEXTURE_2D) {
                        // 2Dのテクスチャの場合

                        // テクスチャのデータをセットする。
                        gl.texImage2D(gl.TEXTURE_2D, 0, internal_format, tex_inf.shape[1], tex_inf.shape[0], 0, format, gl.FLOAT, tex_inf.value); chk();
                    }
                    else {
                        // 3Dのテクスチャの場合

                        assert(dim == gl.TEXTURE_3D, "set-Tex");

                        // テクスチャのデータをセットする。
                        gl.texImage3D(gl.TEXTURE_3D, 0, internal_format, tex_inf.shape[2], tex_inf.shape[1], tex_inf.shape[0], 0, format, gl.FLOAT, tex_inf.value); chk();
                    }
                }
            }
        }


