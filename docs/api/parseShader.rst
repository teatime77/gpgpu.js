parseShader
===========

構文
^^^^^^

parseShader(pkg, param) 

説明
^^^^^^


シェーダのソースコードを解析します。


ソース
^^^^^^

.. code-block:: js

        parseShader(pkg, param) {
            // attribute変数、uniform変数、テクスチャ、varying変数の配列を初期化する。
            pkg.attributes = [];
            pkg.uniforms = [];
            pkg.textures = [];
            pkg.varyings = [];

            // 頂点シェーダとフラグメントシェーダのソースに対し
            for(let shader_text of[ param.vertexShader,  param.fragmentShader ]) {

                // 行ごとに分割する。
                var lines = shader_text.split(/(\r\n|\r|\n)+/);

                // すべての行に対し
                for(let line of lines) {

                    // 行を空白で分割する。
                    var tokens = line.trim().split(/[\s\t]+/);

                    if (tokens.length < 3) {
                        // トークンの長さが3未満の場合
                        continue;
                    }

                    // 最初、2番目、3番目のトークン
                    var tkn0 = tokens[0];
                    var tkn1 = tokens[1];
                    var tkn2 = tokens[2];

                    if (tkn0 != "in" && tkn0 != "uniform" && tkn0 != "out") {
                        // 最初のトークンが in, uniform, out でない場合
                        continue;
                    }

                    if (shader_text == param.fragmentShader && tkn0 != "uniform") {
                        // フラグメントシェーダで uniform でない場合 ( フラグメントシェーダの入力(in)と出力(out)はアプリ側では使わない。 )

                        continue;
                    }
                    assert(tkn1 == "int" || tkn1 == "float" || tkn1 == "vec2" || tkn1 == "vec3" || tkn1 == "vec4" ||
                        tkn1 == "sampler2D" || tkn1 == "sampler3D" ||
                        tkn1 == "mat4" || tkn1 == "mat3" || tkn1 == "bool");


                    var arg_name;
                    var is_array = false;
                    var k1 = tkn2.indexOf("[");
                    if (k1 != -1) {
                        // 3番目のトークンが [ を含む場合

                        // 配列と見なす。
                        is_array = true;

                        // 変数名を得る。
                        arg_name = tkn2.substring(0, k1);
                    }
                    else{
                        // 3番目のトークンが [ を含まない場合

                        var k2 = tkn2.indexOf(";");
                        if (k2 != -1) {
                            // 3番目のトークンが ; を含む場合

                            // 変数名を得る。
                            arg_name = tkn2.substring(0, k2);
                        }
                        else{
                            // 3番目のトークンが ; を含まない場合

                            // 変数名を得る。
                            arg_name = tkn2;
                        }
                    }

                    // 変数の値を得る。
                    var arg_val = param.args[arg_name];

                    if (arg_val == undefined) {
                        if(tokens[0] == "out"){
                            continue;
                        }
                    }

                    if (tkn1 == "sampler2D" || tkn1 == "sampler3D") {
                        // テクスチャのsamplerの場合

                        assert(tokens[0] == "uniform" && arg_val instanceof TextureInfo);

                        // 変数名をセットする。
                        arg_val.name = arg_name;

                        // samplerのタイプをセットする。
                        arg_val.samplerType = tkn1;

                        // 配列かどうかをセットする。
                        arg_val.isArray = is_array;

                        // テクスチャの配列に追加する。
                        pkg.textures.push(arg_val);
                    }
                    else {
                        // テクスチャのsamplerでない場合

                        // 変数の名前、値、型、配列かどうかをセットする。
                        var arg_inf = { name: arg_name, value: arg_val, type: tkn1, isArray: is_array };

                        switch (tokens[0]) {
                            case "in":
                                // attribute変数の場合

                                pkg.attributes.push(arg_inf);
                                break;

                            case "uniform":
                                // uniform変数の場合

                                pkg.uniforms.push(arg_inf);
                                break;

                            case "out":
                                // varying変数の場合

                                pkg.varyings.push(arg_inf);
                                break;
                        }
                    }
                }
            }
        }


