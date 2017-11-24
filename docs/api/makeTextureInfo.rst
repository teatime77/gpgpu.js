makeTextureInfo
===============

構文
^^^^^^

makeTextureInfo(texel_type, shape, value) 

説明
^^^^^^


テクスチャ情報を作ります。


ソース
^^^^^^

.. code-block:: js

        makeTextureInfo(texel_type, shape, value) {
            return new TextureInfo(texel_type, shape, value);
        }


