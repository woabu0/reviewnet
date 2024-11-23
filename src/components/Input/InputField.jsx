function InputField(props) {
  return (
    <div>
      <label className="text-[#777E90] text-[14px] leading-tight m">
        {props.text}
      </label>
      <input
        type={props.types}
        id={props.id}
        name={props.name}
        placeholder={props.placeholder}
        className="w-full h-[48px] xl:h-[40px] 2xl:h-[48px] bg-[#F4F5F6] px-4 rounded-[8px] text-black text-[14px] focus:outline-none"
      />
    </div>
  );
}
export default InputField;
