setUniformsData
===============

構文
^^^^^^

setUniformsData(pkg) 

説明
^^^^^^


uniform変数のデータをセットします。


ソース
^^^^^^

.. code-block:: js

        setUniformsData(pkg) {
            // すべてのuniform変数に対し
            for (let u of pkg.uniforms) {
                if (u.value instanceof Float32Array) {
                    // 値が配列の場合

                    switch (u.type) {
                        case "mat4":
                            gl.uniformMatrix4fv(u.locUniform, false, u.value); chk();
                            break;
                        case "mat3":
                            gl.uniformMatrix3fv(u.locUniform, false, u.value); chk();
                            break;
                        case "vec4":
                            gl.uniform4fv(u.locUniform, u.value); chk();
                            break;
                        case "vec3":
                            gl.uniform3fv(u.locUniform, u.value); chk();
                            break;
                        case "vec2":
                            gl.uniform2fv(u.locUniform, u.value); chk();
                            break;
                        case "float":
                            gl.uniform1fv(u.locUniform, u.value); chk();
                            break;
                        default:
                            assert(false);
                            break;
                    }
                }
                else {
                    // 値が配列でない場合

                    if (u.type == "int" || u.type == "bool") {

                        gl.uniform1i(u.locUniform, u.value); chk();
                    }
                    else {

                        gl.uniform1f(u.locUniform, u.value); chk();
                    }
                }
            }
        }


