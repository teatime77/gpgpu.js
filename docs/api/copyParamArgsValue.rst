copyParamArgsValue
==================

構文
^^^^^^

copyParamArgsValue(param, pkg)

説明
^^^^^^


パラメータの引数の値をコピーします。


ソース
^^^^^^

.. code-block:: js

        copyParamArgsValue(param, pkg){
            for(let args of[ pkg.attributes, pkg.uniforms, pkg.textures, pkg.varyings ]) {
                for (let arg of args) {
                    var val = param.args[arg.name];
                    assert(val != undefined);
                    if (args == pkg.textures) {
                        // テクスチャ情報の場合

                        arg.value = val.value;
                    }
                    else {
                        // テクスチャでない場合

                        arg.value = val;
                    }
                }
            }
        }


