import pymongo
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def plot_results(run_id_1,metrics):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["MemristorNN"]
    mycol = mydb["metrics"]

    for x in mycol.find({},{ "run_id": 1, "name":1,"values":1,"steps":1}):
        if x["run_id"] == run_id_1 and x['name'] == metric:
            plt.plot(x["steps"],x["values"])
    
    for x in mycol.find({},{ "run_id": 1, "name":1,"values":1,"steps":1}):
        if x["run_id"] == run_id_2 and x['name'] == metric:
            plt.plot(x["steps"],x["values"])
    
    plt.xlabel("Training Itertions")  
    plt.ylabel("Test Accuray")
    plt.grid(True,'both')
    plt.legend(['manhatten update rule','manhatten update rule with 3 level input digitization'])
    plt.show()

def plot_Vth(run_id,metrics):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["MemristorNN"]
    mycol = mydb["metrics"]

    for x in mycol.find({},{ "run_id": 1, "name":1,"values":1,"steps":1}):
        if x["run_id"] == run_id and x['name'] in metrics:
            plt.plot(x["steps"],x["values"])
            t = x["values"]
            
    
    
    
    plt.xlabel("Training Cycles")  
    plt.ylabel("Vth [V]")
    plt.grid(True,'both')
    # plt.legend(['manhatten update rule','manhatten update rule with 3 level input digitization'])
    plt.show()
    print()
if __name__ == "__main__":
    # plot_results(25,24,"Test Accuracy (run)")
    # plot_results(25,23,"Test Accuracy (run)")
    plot_Vth(29,['Vth1','Vth2','Vth3','Vth4','Vth5'])
    
    
    

    