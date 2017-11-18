
***************
はじめに
***************

WebGLはブラウザでGPUを使って3D表示をするためのAPIです。
WebGL 2.0は1.0に比べてGPUによる汎用計算(**GPGPU**)の機能が強化されています。

WebGLは複雑な手順に従ってAPIを呼ぶ必要があり使いづらいので、簡単にGPGPUのプログラムを作れるライブラリを作ってみました。

ソースはGitHubに上げています。

https://github.com/teatime77/gpgpu.js/

以下ではこのライブラリの使い方を説明します。

ライブラリの本体は **gpgpu.js** という1個のファイルです。
このファイルをGitHubからコピーしてきて、HTMLファイルのhead部分に以下の1行を入れてください。

.. code-block:: html

    <script type="text/javascript" src="gpgpu.js"></script>



WebGL 2.0は ChromeとFirefox でサポートされていて、WindowsやMacのほか一部のAndroid端末でも動作します。

手持ちの機器でWebGL 2.0が動作するかは以下のページで確認できます。
http://webglreport.com/?v=2
