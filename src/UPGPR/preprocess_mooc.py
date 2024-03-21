import os
import argparse
import pickle

import pandas as pd


def read_relations(dataset, file_name, new_column_names):
    file_path = os.path.join(dataset, file_name)
    print(f"Reading {file_path}")
    df = pd.read_csv(file_path, delimiter="\t", names=new_column_names)
    return df


def read_all_relations(dataset, relations, min_concept_count):
    dataframes = {}
    for relation in relations:
        df = read_relations(
            dataset,
            f"{relation}.json",
            relation.split("-"),
        )
        dataframes[relation] = df

    print(f"Removing concepts in less than {min_concept_count} courses")
    dataframes["course-concept"] = dataframes["course-concept"][
        dataframes["course-concept"].groupby("concept")["concept"].transform("size")
        > min_concept_count
    ]

    return dataframes


def get_enrolments(dataframes, min_user_count):
    print(f"Removing users enrolled in less than {min_user_count} courses")
    enrolments = dataframes["user-course"][
        dataframes["user-course"].course.isin(dataframes["school-course"].course)
    ]
    enrolments = enrolments[
        enrolments.groupby("user")["user"].transform("size") >= min_user_count
    ]
    return enrolments


def get_all_entities(dataframes, enrolments):
    print(f"Extracting entities")
    entities = {}
    entities["users"] = enrolments.user.unique()
    entities["courses"] = enrolments.course.unique()
    entities["teachers"] = dataframes["teacher-course"][
        dataframes["teacher-course"].course.isin(entities["courses"])
    ].teacher.unique()

    for i, t in enumerate(entities["teachers"]):
        entities["teachers"][i] = t.replace(" ", "_")

    entities["schools"] = dataframes["school-course"][
        dataframes["school-course"].course.isin(entities["courses"])
    ].school.unique()

    entities["concepts"] = dataframes["course-concept"].concept.unique()

    for entity in entities:
        print(f"Number of {entity}: {len(entities[entity])}")

    return entities


def save_entity(entity, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(entity)
        f.write(out)


def save_entities(save_dir, entities):
    for entity in entities:
        file_name = os.path.join(save_dir, f"{entity}.txt")
        save_entity(entities[entity], file_name)


def get_entity_to_idx(entity):
    entity_to_idx = {}
    for i, e in enumerate(entity):
        entity_to_idx[e] = i
    return entity_to_idx


def get_all_entities_to_idx(entities):
    entities_to_idx = {}
    for entity in entities:
        entities_to_idx[entity] = get_entity_to_idx(entities[entity])
    return entities_to_idx


def save_enrolments(save_dir, enrolments, entities_to_idx):
    # enr_by_user = {}
    out = []
    for idx in enrolments.index:
        u = enrolments.user[idx]
        c = enrolments.course[idx]
        u_idx = entities_to_idx["users"][u]
        c_idx = entities_to_idx["courses"][c]
        # enr_by_user[u_idx] = enr_by_user.get(u_idx, []) + [c_idx]
        out.append(f"{u_idx} {c_idx}")

    file_name = os.path.join(save_dir, "enrolments.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)

    # pkl_file_name = os.path.join(save_dir, "enrolments.pkl")
    # with open(pkl_file_name, "wb") as f:
    #     pickle.dump(enr_by_user, f)


def save_all_relations(save_dir, dataframes, entities, entities_to_idx):
    course_to_school = {}
    course_to_teachers = {}
    course_to_concepts = {}

    for idx in dataframes["school-course"].index:
        s = dataframes["school-course"].school[idx]
        c = dataframes["school-course"].course[idx]
        course_to_school[c] = s

    out = []
    for course in entities["courses"]:
        out.append(str(entities_to_idx["schools"][course_to_school[course]]))

    file_name = os.path.join(save_dir, "course_school.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)

    for idx in dataframes["teacher-course"].index:
        t = dataframes["teacher-course"].teacher[idx]
        c = dataframes["teacher-course"].course[idx]
        t = t.replace(" ", "_")
        course_to_teachers[c] = course_to_teachers.get(c, []) + [t]

    out = []
    for course in entities["courses"]:
        ts = course_to_teachers.get(course, [])
        ts = [str(entities_to_idx["teachers"][t]) for t in ts]
        out.append(" ".join(ts))

    file_name = os.path.join(save_dir, "course_teachers.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)

    for idx in dataframes["course-concept"].index:
        k = dataframes["course-concept"].concept[idx]
        c = dataframes["course-concept"].course[idx]
        course_to_concepts[c] = course_to_concepts.get(c, []) + [k]

    out = []

    for course in entities["courses"]:
        cs = course_to_concepts.get(course, [])
        cs = [str(entities_to_idx["concepts"][c]) for c in cs]
        out.append(" ".join(cs))

    file_name = os.path.join(save_dir, "course_concepts.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/mooc/MOOCCube/relations")
    parser.add_argument("--save_dir", type=str, default="data/mooc/processed_files")
    parser.add_argument("--min_concept_count", type=int, default=10)
    parser.add_argument("--min_user_count", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    relations = [
        "course-concept",
        "school-course",
        "teacher-course",
        "user-course",
    ]

    dataframes = read_all_relations(args.dataset, relations, args.min_concept_count)

    enrolments = get_enrolments(dataframes, args.min_user_count)

    entities = get_all_entities(dataframes, enrolments)

    save_entities(args.save_dir, entities)

    entities_to_idx = get_all_entities_to_idx(entities)

    save_enrolments(args.save_dir, enrolments, entities_to_idx)

    save_all_relations(args.save_dir, dataframes, entities, entities_to_idx)


if __name__ == "__main__":
    main()
