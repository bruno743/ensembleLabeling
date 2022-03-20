import sys

def atrib_rotul(grupos_disc, attr_classe, infor_attrs, variacao, disc_infor):
    rotulos = []
    # para cada grupo, aciona o método de geração de rótulos
    for grupo in grupos_disc:
        clt = grupo[attr_classe].unique()[0]
        at_info = [i[1] for i in infor_attrs if i[0]==clt][0]
        result_rot = defin_rotul(grupo.drop([attr_classe], axis=1), at_info, disc_infor, variacao)
        rotulos.append((clt, result_rot))
    return rotulos

def defin_rotul(cluster, at_info, disc_infor, variacao):
    medias = [(i, acuracia*100) for i, acuracia in at_info]
    medias.sort(key=lambda x: x[1], reverse=True)
    minn = medias[0][1] - variacao
    titulos = cluster.columns.values.tolist()
    
    result = []
    # para cada atributo
    for i in range(cluster.shape[1]):
        # se a acuracia do atributo for superior a um valor,
        # será "calculado" o intervalo e o par atributo-intervalo fará parte do rótulo
        if medias[i][1] >= minn: 
            attr = medias[i][0]
            info = [j[1] for j in disc_infor if j[0]==attr][0]
            most_comun_value = cluster[attr].mode()[0]
            try:
                rotulo = (attr, round(info[most_comun_value], 2), round(info[most_comun_value+1], 2))
            except:  
                print("Não foi possivel atribuir rotulo aos clusters")
                #Isso ocorre pois algum atributo foi discretizado de forma 
                #incorreta(provavelmente um atributo tem um mesmo valor 
                #para todas as entradas na base de dados). Eliminar esse 
                #atributo da base pode ser uma solucao para o problema
                input("Pressione ENTER para sair")
                sys.exit()
            result.append(rotulo)
    return result