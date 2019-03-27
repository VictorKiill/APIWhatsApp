# APIWhatsApp

**Para que serve?**
>Essa Bliblioteca de Python serve para que uma mensagem pode ser enviada para um qualquer número que esteja cadastrado no WhatsApp, independente se tal número seja adicionado ou não aos contatos de quem está a enviando.

**Por que eu usaria tal API?**
>Caso um cliente queria dar um número de contato do WhatsApp da empresa no site, aplicativo ou outros meios na internet, basta gerar o link de contato para tirar a necessidade do usuário de digitar ou lembrar desse número.
>Outras formas de usar a API é de gerar um link para contato quando a necessidade é mostrar o contado de várias pessoas para várias pessoas, como em sites e aplicativos de relacionamento.

**Como funciona?**
>Basta importar o arquivo apiWhatsApp.py no seu código, chamar a função enviarMensagem(num, msg), substituir "num" pelo número de celular do remetente e "msg" pela mensagem a ser enviada, depois de chamada a função retornará o link para enviar a mensagem. O link retornado redireciona o usuário à uma página da api do whatsapp e caso send seja clicado o WhatsApp Web será aberto, ou o aplicativo do WhatsApp caso o link seja aberto no smartphone, onde o mesmo poderá iniciar uma conversa com o remetente já com a mensagem pronta para o envio.

**E se eu quiser apenas testar a API sem ter que criar um novo código?**
>Basta iniciar o client.py, informar o que é pedido e o link será gerado.
