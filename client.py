from apiWhatsApp import enviarMensagem

def main():
    while True:
        num1 = input("Digite o número código internacional do número: ")
        num2 = input("Digite o número DD do número: ")
        num3 = input("Digite o número: ")
        if not num1.isdigit() or not num2.isdigit() or not num3.isdigit():
            print("Número incorreto!")
            print("")
        else:
            break

    num = num1 + num2 + num3
    print("")

    msg = input("Digite a mensagem para ser enviada(Não é obrigatório): ")

    print(enviarMensagem(num, msg))

main()
