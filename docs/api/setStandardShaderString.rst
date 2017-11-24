setStandardShaderString
=======================

構文
^^^^^^

setStandardShaderString() 

説明
^^^^^^


標準のシェーダの文字列をセットします。


ソース
^^^^^^

.. code-block:: js

        setStandardShaderString() {
            this.textureSphereVertexShader = `

                const vec3 uAmbientColor = vec3(0.2, 0.2, 0.2);
                const vec3 uLightingDirection =  normalize( vec3(0.25, 0.25, 1) );
                const vec3 uDirectionalColor = vec3(0.8, 0.8, 0.8);

                // 位置
                in vec3 VertexPosition;

                // 法線
                in vec3 VertexNormal;

                // テクスチャ座標
                in vec2 TextureCoord;

                uniform mat4 uPMVMatrix;
                uniform mat3 uNMatrix;

                out vec3 vLightWeighting;

                out vec2 uv0;
                out vec2 uv1;

                void main(void) {
                    gl_Position = uPMVMatrix * vec4(VertexPosition, 1.0);

                    vec3 transformedNormal = uNMatrix * VertexNormal;
                    float directionalLightWeighting = max(dot(transformedNormal, uLightingDirection), 0.0);
                    vLightWeighting = uAmbientColor +uDirectionalColor * directionalLightWeighting;

                    uv0 = fract( TextureCoord.st );
                    uv1 = fract( TextureCoord.st + vec2(0.5,0.5) ) - vec2(0.5,0.5);
                }
            `;

            // GPGPU用のフラグメントシェーダ。(何も処理はしない。)
            this.minFragmentShader =
               `out vec4 color;

                void main(){
                    color = vec4(1.0);
                }`;

            // デフォルトの動作のフラグメントシェーダ
            this.defaultFragmentShader =
               `in vec3 vLightWeighting;
                in vec2 uv0;
                in vec2 uv1;

                uniform sampler2D TextureImage;

                out vec4 color;

                void main(void) {
                    vec2 uvT;

                    uvT.x = ( fwidth( uv0.x ) < fwidth( uv1.x )-0.001 ) ? uv0.x : uv1.x ;
                    uvT.y = ( fwidth( uv0.y ) < fwidth( uv1.y )-0.001 ) ? uv0.y : uv1.y ;

                    vec4 textureColor = texture(TextureImage, uvT);

                    color = vec4(textureColor.rgb * vLightWeighting, textureColor.a);
                }
                `;
        }


