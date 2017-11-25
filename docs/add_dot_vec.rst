
ベクトルの加算と内積
====================

前回の :doc:`add_mul_vec` を少し変えて、ベクトルの加算と内積をしてみます。

.. image:: _static/img/AddDotVec.png



入力変数のA,Bと出力変数のCは **vec3** として宣言します。

Cの計算式は同じですが、今回は **vec3** での加算です。
Dの計算式は内積( **dot** )になります。

頂点シェーダのコード
^^^^^^^^^^^^^^^^^^^^

.. code-block:: glsl

    // 入力変数A
    in  vec3 A;

    // 入力変数B
    in  vec3 B;

    // 出力変数C
    out vec3 C;

    // 出力変数D
    out float D;

    // 要素ごとに呼ばれる関数。
    void main(void ) {
        C = A + B;
        D = dot(A, B);
    }


ここでも **入力変数と出力変数の要素の数はすべて同じ** という原則は守られています。
A, B, Cは **vec3** の要素が2個で、Dは **float** の要素が2個。
つまりA, B, C, Dはすべて2個の要素を持っています。


サンプルのURL
    http://lkzf.info/gpgpu.js/samples/AddDotVec.html
