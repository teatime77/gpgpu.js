コールグラフ
==============

.. blockdiag::

    blockdiag {
        makePackage -> parseShader;
        makePackage -> makeShader;
        makePackage -> makeProgram;
        makePackage -> setUniformLocation;
        makePackage -> makeTexture;
        makePackage -> makeAttrib
        makePackage -> makeVertexIndexBuffer;

        compute -> makePackage;
        compute -> copyParamArgsValue;
        compute -> setAttribData;
        compute -> setTextureData;
        compute -> setUniformsData;
    }

.. 
    .. digraph:: testname

        node[fontname=ipag]
        "bar" -> "baz" -> "日本語"

        "GPGPU.constructor" -> setStandardShaderString


    .. graphviz::

       digraph foo {
          "bar" -> "baz";
       }
