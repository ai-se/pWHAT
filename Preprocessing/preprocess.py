from __future__ import division


def preprocessing(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["column"] != "Configuration" and data.attrib["column"] != "Variable Features":
                d[data.attrib["column"]] = float(data.text.replace(",", "."))
            elif data.attrib["column"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["column"]] = temp_dict
            else:
                d[data.attrib["column"]] = [dd for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)

    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    all_variable_features = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])
        all_variable_features.extend(C["Variable Features"].keys())

    b_configuration_names = list(set(all_configuratio_names))
    variable_feature_names = list(set(all_variable_features))

    mix = b_configuration_names + variable_feature_names

    configuraiton_dict = {mix[x]:x for x in xrange(len(mix))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    configuration_names += ["$"+s for s in variable_feature_names]
    performance_name = "$<Measured Value"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []
    #
    # import pdb
    # pdb.set_trace()

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1
        for i in xrange(len(variable_feature_names)):
            # print i, len(decisions), len(configuration_names)
            decisions[len(b_configuration_names) + i] = c["Variable Features"][variable_feature_names[i]]
        objective = str(c["Measured Value"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()

def preprocessing_AJStats(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)


    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]:x for x in xrange(len(configuration_names))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<AnalysisTime"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["AnalysisTime"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_Apache(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)


    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]:x for x in xrange(len(configuration_names))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<ResponseRate"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["ResponseRate"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_BerkeleyC(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                print row
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)


    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]:x for x in xrange(len(configuration_names))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<BinarySize"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["BinarySize"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_BerkeleyDB(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)


    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]:x for x in xrange(len(configuration_names))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<MainMemory"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["MainMemory"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_BerkeleyDBC(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)


    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]:x for x in xrange(len(configuration_names))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<Performance"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["Performance"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_BerkeleyDBJ(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)


    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]:x for x in xrange(len(configuration_names))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<Performance"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["Performance"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_clasp(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["column"] != "Configuration":
                d[data.attrib["column"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["column"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)


    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]:x for x in xrange(len(configuration_names))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<SolvingTime"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["SolvingTime"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_Dune(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration" and data.attrib["columname"] != "Variable Features":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            elif data.attrib["columname"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["columname"]] = temp_dict
            else:
                d[data.attrib["columname"]] = [dd for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    all_variable_features = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])
        all_variable_features.extend(C["Variable Features"].keys())

    b_configuration_names = list(set(all_configuratio_names))
    variable_feature_names = list(set(all_variable_features))

    mix = b_configuration_names + variable_feature_names

    configuraiton_dict = {mix[x]: x for x in xrange(len(mix))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    configuration_names += ["$" + s for s in variable_feature_names]
    performance_name = "$<Performance"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []
    #
    # import pdb
    # pdb.set_trace()

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1
        for i in xrange(len(variable_feature_names)):
            # print i, len(decisions), len(configuration_names)
            decisions[len(b_configuration_names) + i] = c["Variable Features"][variable_feature_names[i]]
        objective = str(c["Performance"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_Elevator(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<Performance"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["Performance"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_Email(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<Performance"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["Performance"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_EPL(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<BinarySize"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["BinarySize"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_Hipacc(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration" and data.attrib["columname"] != "Variable Features":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            elif data.attrib["columname"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["columname"]] = temp_dict
            else:
                d[data.attrib["columname"]] = [dd for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    all_variable_features = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])
        all_variable_features.extend(C["Variable Features"].keys())

    b_configuration_names = list(set(all_configuratio_names))
    variable_feature_names = list(set(all_variable_features))

    mix = b_configuration_names + variable_feature_names

    configuraiton_dict = {mix[x]: x for x in xrange(len(mix))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    configuration_names += ["$" + s for s in variable_feature_names]
    performance_name = "$<Performance"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []
    #
    # import pdb
    # pdb.set_trace()

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1
        for i in xrange(len(variable_feature_names)):
            # print i, len(decisions), len(configuration_names)
            decisions[len(b_configuration_names) + i] = c["Variable Features"][variable_feature_names[i]]
        objective = str(c["Performance"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_HSMGP_num(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration" and data.attrib["columname"] != "Variable Features":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            elif data.attrib["columname"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["columname"]] = temp_dict
            else:
                d[data.attrib["columname"]] = [dd for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    all_variable_features = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])
        all_variable_features.extend(C["Variable Features"].keys())

    b_configuration_names = list(set(all_configuratio_names))
    variable_feature_names = list(set(all_variable_features))

    mix = b_configuration_names + variable_feature_names

    configuraiton_dict = {mix[x]: x for x in xrange(len(mix))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    configuration_names += ["$" + s for s in variable_feature_names]
    performance_name = "$<AverageTimePerIteration"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []
    #
    # import pdb
    # pdb.set_trace()

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1
        for i in xrange(len(variable_feature_names)):
            # print i, len(decisions), len(configuration_names)
            decisions[len(b_configuration_names) + i] = c["Variable Features"][variable_feature_names[i]]
        objective = str(c["AverageTimePerIteration"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_JavaGC(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["column"] != "Configuration" and data.attrib["column"] != "Variable Features":
                d[data.attrib["column"]] = float(data.text.replace(",", "."))
            elif data.attrib["column"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["column"]] = temp_dict
            else:
                d[data.attrib["column"]] = [dd for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    all_variable_features = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])
        all_variable_features.extend(C["Variable Features"].keys())

    b_configuration_names = list(set(all_configuratio_names))
    variable_feature_names = list(set(all_variable_features))

    mix = b_configuration_names + variable_feature_names

    configuraiton_dict = {mix[x]: x for x in xrange(len(mix))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    configuration_names += ["$" + s for s in variable_feature_names]
    performance_name = "$<Measured Value"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []
    #
    # import pdb
    # pdb.set_trace()

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1
        for i in xrange(len(variable_feature_names)):
            # print i, len(decisions), len(configuration_names)
            decisions[len(b_configuration_names) + i] = c["Variable Features"][variable_feature_names[i]]
        objective = str(c["Measured Value"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_LinkedList(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<BinarySize"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["BinarySize"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_lrzip(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["column"] != "Configuration":
                d[data.attrib["column"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["column"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<CompressionTime"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["CompressionTime"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_PKJab(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<BinarySize"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["BinarySize"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_PrevaylerPP(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<BinarySize"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["BinarySize"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_SQLite(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for i, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<MainMemory"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["MainMemory"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_TriMesh(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration" and data.attrib["columname"] != "Variable Features":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            elif data.attrib["columname"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["columname"]] = temp_dict
            else:
                d[data.attrib["columname"]] = [dd for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)

    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    all_variable_features = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])
        all_variable_features.extend(C["Variable Features"].keys())

    b_configuration_names = list(set(all_configuratio_names))
    variable_feature_names = list(set(all_variable_features))

    mix = b_configuration_names + variable_feature_names

    configuraiton_dict = {mix[x]:x for x in xrange(len(mix))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    configuration_names += ["$"+s for s in variable_feature_names]
    performance_name = "$<NumberIterations, $<TimeToSolution , $<TimePerIteration"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []


    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1
        for i in xrange(len(variable_feature_names)):
            # print i, len(decisions), len(configuration_names)
            decisions[len(b_configuration_names) + i] = c["Variable Features"][variable_feature_names[i]]
        objective1 = str(c["NumberIterations"])
        objective2 = str(c["TimeToSolution"])
        objective3 = str(c["TimePerIteration"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective1+ "," + objective2+ "," + objective3)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_WGet(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for row in rows:
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            else:
                d[data.attrib["columname"]] = [dd.strip() for dd in data.text.strip().split(",") if len(dd) != 0]
        Content.append(d)

    assert (len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content: all_configuratio_names.extend(C['Configuration'])

    configuration_names = [c.strip() for c in list(set(all_configuratio_names))]
    configuration_dict = {configuration_names[x]: x for x in xrange(len(configuration_names))}
    configuration_names = ["$" + s for s in set(all_configuratio_names)]
    performance_name = "$<MainMemory"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []

    for c in Content:
        active_configurations = c["Configuration"]
        active_indexes = [configuration_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(configuration_names))]
        for ai in active_indexes: decisions[ai] = 1
        objective = str(c["MainMemory"])
        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_x264(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration" and data.attrib["columname"] != "Variable Features":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            elif data.attrib["columname"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["columname"]] = temp_dict
            else:
                d[data.attrib["columname"]] = [dd for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)

    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])

    b_configuration_names = list(set(all_configuratio_names))

    mix = b_configuration_names

    configuraiton_dict = {mix[x]:x for x in xrange(len(mix))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<MainMemory, $<Performance"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []


    for cc, c in enumerate(Content):
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1
        try: objective1 = str(c["MainMemory"])
        except: objective1 = "-1"
        try:
            objective2 = str(c["Performance"])
        except:objective2 = "-1"

        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective1+ "," + objective2)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_x264_DB(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration" and data.attrib["columname"] != "Variable Features":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            elif data.attrib["columname"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["columname"]] = temp_dict
            else:
                d[data.attrib["columname"]] = [dd for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)

    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])

    b_configuration_names = list(set(all_configuratio_names))

    mix = b_configuration_names

    configuraiton_dict = {mix[x]:x for x in xrange(len(mix))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<PSNR, $<Energy, $<SSIM, $<Time, $<Watt, $<Speed, $<Size"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []


    for cc, c in enumerate(Content):
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1

        try: objective1 = str(c["PSNR"])
        except: objective1 = "-1"

        try:
            objective2 = str(c["Energy"])
        except:objective2 = "-1"

        try: objective3 = str(c["SSIM"])
        except: objective3 = "-1"

        try:
            objective4 = str(c["Time"])
        except:objective4 = "-1"

        try: objective5 = str(c["Watt"])
        except: objective5 = "-1"

        try:
            objective6 = str(c["Speed"])
        except:objective6 = "-1"

        try: objective7 = str(c["Size"])
        except: objective7 = "-1"


        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective1+ "," + objective2 +
                     "," + objective3+ "," + objective4+ "," + objective5+ "," + objective6+ "," + objective7)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()


def preprocessing_ZipMe(fname):
    Measurements = "./Measurements/"
    filename = Measurements + fname + "/measurements.xml"
    Content = []
    from xml.etree import ElementTree
    root = ElementTree.parse(filename).getroot()
    rows = root.getchildren()
    for lineno, row in enumerate(rows):
        datas = row.getchildren()
        d = {}
        for data in datas:
            if data.attrib["columname"] != "Configuration" and data.attrib["columname"] != "Variable Features":
                d[data.attrib["columname"]] = float(data.text.replace(",", "."))
            elif data.attrib["columname"] == "Variable Features":
                text = data.text.strip()
                temp_vars = text.split(",")
                temp_dict = {}
                for temp_var in temp_vars:
                    key, value = temp_var.split(";")
                    temp_dict[key] = int(value)
                d[data.attrib["columname"]] = temp_dict
            else:
                d[data.attrib["columname"]] = [dd for dd in data.text.strip().split(",") if len(dd)!=0]
        Content.append(d)

    assert(len(rows) == len(Content)), "Something is wrong"

    all_configuratio_names = []
    for C in Content:
        all_configuratio_names.extend(C['Configuration'])

    b_configuration_names = list(set(all_configuratio_names))

    mix = b_configuration_names

    configuraiton_dict = {mix[x]:x for x in xrange(len(mix))}
    configuration_names = ["$"+s for s in set(all_configuratio_names)]
    performance_name = "$<Performance, $<BinarySearch"
    header = ",".join(configuration_names) + "," + performance_name
    lines = []


    for cc, c in enumerate(Content):
        active_configurations = c["Configuration"]
        active_indexes = [configuraiton_dict[ac] for ac in active_configurations]
        decisions = [0 for _ in xrange(len(mix))]
        for ai in active_indexes: decisions[ai] = 1
        try: objective1 = str(c["Performance"])
        except: objective1 = "-1"
        try:
            objective2 = str(c["BinarySearch"])
        except:objective2 = "-1"

        lines.append(",".join([str(decision) for decision in decisions]) + "," + objective1+ "," + objective2)

    f = open("./CSV/" + fname + ".csv", "w")
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
    f.close()



# preprocessing_AJStats("AJStats")
# preprocessing_Apache("Apache")
# preprocessing_BerkeleyC("BerkeleyC")
# preprocessing_BerkeleyDB("BerkeleyDB")
# preprocessing_BerkeleyDBC("BerkeleyDBC")
# preprocessing_BerkeleyDBJ("BerkeleyDBJ")
# preprocessing_clasp("clasp")
# preprocessing_Dune("Dune")
# preprocessing_Email("Email")
# preprocessing_EPL("EPL")
# preprocessing_Hipacc("Hipacc")
# preprocessing_HSMGP_num("HSMGP_num")
# preprocessing_lrzip("lrzip")
# preprocessing_PKJab("PKJab")
# preprocessing_SQLite("SQLite")
# preprocessing_TriMesh("TriMesh")
# preprocessing_WGet("WGet")
# preprocessing_x264_DB("x264-DB")
preprocessing_ZipMe("ZipMe")