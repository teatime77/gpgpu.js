setUniformLocation
==================

構文
^^^^^^

setUniformLocation(pkg) 

説明
^^^^^^


ユニフォーム変数のロケーションをセットします。


ソース
^^^^^^

.. code-block:: js

        setUniformLocation(pkg) {
            pkg.uniforms.forEach(u => u.locUniform = gl.getUniformLocation(pkg.program, u.name), chk());
        }


